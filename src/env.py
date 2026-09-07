"""Captura do ambiente de execucao (para a tabela de hardware/software do artigo)."""
from __future__ import annotations

import json
import os
import platform
import subprocess

import torch
import torchvision


def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return None


def collect_environment(cpu_threads: int | None = None) -> dict:
    cuda = torch.cuda.is_available()
    info = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_cores": os.cpu_count(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_total_mem_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1) if cuda else None,
        "gpu_driver": _nvidia_smi("driver_version") if cuda else None,
        "torch_num_threads_for_cpu_bench": cpu_threads,
    }
    return info


def write_environment(path: str, cpu_threads: int | None = None) -> dict:
    info = collect_environment(cpu_threads)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"[env] ambiente salvo em {path}")
    return info
