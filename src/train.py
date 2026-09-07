"""Treina UM modelo. Resumivel: salva checkpoint a cada epoca e continua de onde parou.

Uso:
    python -m src.train --model resnet50 --data /caminho/isic2019 --epochs 20
    python -m src.train --model mobilenet_v2               # retoma se houver checkpoint
    python -m src.train --model resnet50 --fresh           # ignora checkpoint existente
"""
from __future__ import annotations

import argparse
import math
import os
import time

import pandas as pd
import torch
import torch.nn as nn

from .data import build_dataloaders
from .efficiency import GpuEnergyMeter
from .engine import evaluate, train_one_epoch
from .metrics import compute_metrics
from .models import build_model, display_name
from .utils import ensure_dir, get_device, load_config, resolve_data_dir, set_seed

HISTORY_COLUMNS = [
    "epoch", "lr", "train_loss", "val_loss", "val_accuracy",
    "val_balanced_accuracy", "val_f1_macro", "val_auc_macro_ovr",
    "epoch_time_s", "peak_vram_mb", "gpu_energy_wh",
]


def build_optimizer(model, cfg):
    t = cfg["train"]
    if t["optimizer"].lower() != "adamw":
        raise ValueError(f"optimizer nao suportado: {t['optimizer']}")
    return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])


def build_scheduler(optimizer, cfg):
    t = cfg["train"]
    if t["scheduler"] == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _e: 1.0)
    warmup, total = t["warmup_epochs"], t["epochs"]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=None, help="diretorio do ISIC 2019 (senao baixa via kagglehub)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="ignora checkpoint e comeca do zero")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides={"train.epochs": args.epochs})
    set_seed(cfg["seed"], cfg["deterministic"])
    device = get_device()

    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    ckpt_dir = ensure_dir(os.path.join(cfg["paths"]["checkpoints_dir"], args.model))
    data_dir = resolve_data_dir(cfg, args.data)

    print(f"\n=== {display_name(args.model)} | device={device} | epocas={cfg['train']['epochs']} ===")

    loaders, class_weights, class_names, _ = build_dataloaders(cfg, data_dir)
    num_classes = len(class_names)

    model = build_model(args.model, num_classes, pretrained=True).to(device)
    weight = class_weights.to(device) if cfg["train"]["class_weighting"] else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    except TypeError:  # torch antigo
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history_path = os.path.join(results_dir, f"history_{args.model}.csv")
    last_ckpt = os.path.join(ckpt_dir, "last.pt")
    best_ckpt = os.path.join(ckpt_dir, "best.pt")

    start_epoch, best_f1, epochs_no_improve = 0, -1.0, 0
    history: list[dict] = []

    if last_ckpt and os.path.exists(last_ckpt) and not args.fresh:
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", -1.0)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        if os.path.exists(history_path):
            history = pd.read_csv(history_path).to_dict("records")
        print(f"[resume] retomando da epoca {start_epoch} (melhor macro-F1={best_f1:.4f})")

    if start_epoch >= cfg["train"]["epochs"]:
        print("[train] numero de epocas ja atingido; nada a fazer.")
        return

    patience = cfg["train"]["early_stopping_patience"]

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        lr_now = optimizer.param_groups[0]["lr"]
        t0 = time.time()

        meter = GpuEnergyMeter() if cfg["train"]["log_gpu_energy"] else None
        if meter is not None:
            meter.__enter__()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer,
                                     device, scaler, use_amp)
        if meter is not None:
            meter.__exit__()

        val_loss, y_true, y_pred, y_prob = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        m = compute_metrics(y_true, y_pred, y_prob, class_names)
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": m["accuracy"],
            "val_balanced_accuracy": m["balanced_accuracy"],
            "val_f1_macro": m["f1_macro"],
            "val_auc_macro_ovr": m["auc_macro_ovr"],
            "epoch_time_s": time.time() - t0,
            "peak_vram_mb": peak_vram,
            "gpu_energy_wh": (meter.energy_wh if (meter is not None and meter.available) else float("nan")),
        }
        history.append(row)
        pd.DataFrame(history).reindex(columns=HISTORY_COLUMNS).to_csv(history_path, index=False)

        print(f"epoca {epoch + 1:>2}/{cfg['train']['epochs']} | "
              f"lr {lr_now:.2e} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"acc {m['accuracy']:.3f} | macro-F1 {m['f1_macro']:.4f} | "
              f"{row['epoch_time_s']:.0f}s | VRAM {peak_vram:.0f}MB")

        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if use_amp else None,
            "best_f1": best_f1, "epochs_no_improve": epochs_no_improve,
            "model_name": args.model, "class_names": class_names, "config": cfg,
        }
        torch.save(state, last_ckpt)

        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            epochs_no_improve = 0
            state["best_f1"] = best_f1
            torch.save(state, best_ckpt)
            print(f"   novo melhor macro-F1 = {best_f1:.4f}  -> {best_ckpt}")
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"[early stopping] {patience} epocas sem melhora.")
                break

    print(f"\n[train] concluido. Melhor macro-F1 (val) = {best_f1:.4f}")
    print(f"[train] historico: {history_path}")
    print(f"[train] melhor checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
