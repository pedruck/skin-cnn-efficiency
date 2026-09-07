"""Metricas de custo/eficiencia de hardware, intrinsecas a arquitetura.

Cada medida usa uma instancia recem-criada do modelo (pesos aleatorios; os valores
nao afetam FLOPs nem latencia) para evitar contaminacao entre medidas -- em
particular, ``thop.profile`` injeta buffers no modelo.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .models import INPUT_SIZE, build_model


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model, tmp_path: str = "temp_size.p") -> float:
    torch.save(model.state_dict(), tmp_path)
    size = os.path.getsize(tmp_path) / 1024**2
    os.remove(tmp_path)
    return size


def compute_macs(name: str, num_classes: int, device: torch.device) -> float:
    """MACs (multiply-accumulate) para uma imagem. FLOPs ~= 2 x MACs."""
    from thop import profile

    model = build_model(name, num_classes, pretrained=False).to(device).eval()
    dummy = torch.randn(INPUT_SIZE, device=device)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(macs)


@torch.no_grad()
def measure_latency(name: str, num_classes: int, device: torch.device,
                    warmup: int = 20, runs: int = 200,
                    batch_size: int = 1) -> dict:
    """Latencia por inferencia (batch pequeno) ou throughput em lote."""
    model = build_model(name, num_classes, pretrained=False).to(device).eval()
    shape = (batch_size, *INPUT_SIZE[1:])
    dummy = torch.randn(shape, device=device)

    for _ in range(warmup):
        model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        times_ms = np.empty(runs, dtype=np.float64)
        for i in range(runs):
            starter.record()
            model(dummy)
            ender.record()
            torch.cuda.synchronize()
            times_ms[i] = starter.elapsed_time(ender)
    else:
        times_ms = np.empty(runs, dtype=np.float64)
        for i in range(runs):
            t0 = time.perf_counter()
            model(dummy)
            times_ms[i] = (time.perf_counter() - t0) * 1000.0

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_batch = times_ms
    per_image = times_ms / batch_size
    return {
        "batch_size": batch_size,
        "latency_ms_mean": float(per_image.mean()),
        "latency_ms_std": float(per_image.std()),
        "latency_ms_median": float(np.median(per_image)),
        "latency_ms_p95": float(np.percentile(per_image, 95)),
        "throughput_img_s": float(batch_size * 1000.0 / per_batch.mean()),
    }


def profile_model(name: str, num_classes: int, device: torch.device, eff_cfg: dict) -> dict:
    """Consolida todas as metricas de eficiencia de um modelo numa linha."""
    ref = build_model(name, num_classes, pretrained=False)
    total, trainable = count_parameters(ref)
    size = model_size_mb(ref)
    del ref

    macs = compute_macs(name, num_classes, device)

    row = {
        "model": name,
        "params_total_m": total / 1e6,
        "params_trainable_m": trainable / 1e6,
        "size_mb": size,
        "gmacs": macs / 1e9,
        "gflops_approx": 2 * macs / 1e9,
    }

    warmup, runs = eff_cfg["latency_warmup"], eff_cfg["latency_runs"]

    if device.type == "cuda":
        gpu = measure_latency(name, num_classes, device, warmup, runs, batch_size=1)
        row["gpu_latency_ms_mean"] = gpu["latency_ms_mean"]
        row["gpu_latency_ms_std"] = gpu["latency_ms_std"]
        row["gpu_latency_ms_p95"] = gpu["latency_ms_p95"]
        row["gpu_throughput_img_s"] = gpu["throughput_img_s"]
        batched = measure_latency(name, num_classes, device, warmup, max(runs // 4, 20),
                                  batch_size=eff_cfg["batched_batch_size"])
        row["gpu_batched_throughput_img_s"] = batched["throughput_img_s"]

    # CPU: fixa o numero de threads para tornar a medida comparavel
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(int(eff_cfg["cpu_threads"]))
    cpu = measure_latency(name, num_classes, torch.device("cpu"),
                          warmup=max(warmup // 2, 5), runs=max(runs // 4, 20), batch_size=1)
    torch.set_num_threads(prev_threads)
    row["cpu_threads"] = int(eff_cfg["cpu_threads"])
    row["cpu_latency_ms_mean"] = cpu["latency_ms_mean"]
    row["cpu_latency_ms_std"] = cpu["latency_ms_std"]
    row["cpu_latency_ms_p95"] = cpu["latency_ms_p95"]
    row["cpu_throughput_img_s"] = cpu["throughput_img_s"]

    return row


# --------------------------------------------------------------------------- #
# Medidor opcional de energia da GPU durante o treino (amostra a potencia)
# --------------------------------------------------------------------------- #
class GpuEnergyMeter:
    """Integra potencia (W) da GPU ao longo do tempo -> energia (Wh).

    Silenciosamente vira no-op se ``nvidia-ml-py`` (pynvml) nao estiver instalado
    ou nao houver GPU.
    """

    def __init__(self, sample_hz: float = 5.0):
        self.interval = 1.0 / sample_hz
        self.energy_wh = 0.0
        self._ok = False
        self._thread = None
        self._stop = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._ok = True
        except Exception:
            self._pynvml = None

    def _run(self):
        last = time.perf_counter()
        while not self._stop:
            time.sleep(self.interval)
            now = time.perf_counter()
            try:
                watts = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self.energy_wh += watts * (now - last) / 3600.0
            except Exception:
                pass
            last = now

    def __enter__(self):
        if self._ok:
            import threading

            self._stop = False
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._ok:
            self._stop = True
            if self._thread is not None:
                self._thread.join(timeout=2)

    @property
    def available(self) -> bool:
        return self._ok
