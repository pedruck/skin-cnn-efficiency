"""Avaliacao final no conjunto de TESTE + benchmark de eficiencia.

Le os checkpoints ``best.pt`` de cada modelo, avalia no test set (guardando as
predicoes) e mede as metricas de custo de hardware. Escreve em ``results/``:

    environment.json           ambiente de execucao
    performance.csv            metricas preditivas por modelo
    efficiency.csv             metricas de custo por modelo
    summary.csv                juncao + colunas de custo-beneficio
    per_class_<model>.csv      precisao/recall/F1 por classe
    test_predictions_<model>.npz   y_true, y_pred, y_prob  (regenera qualquer figura)

Uso:
    python -m src.benchmark --data /caminho/isic2019
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import build_dataloaders
from .efficiency import profile_model
from .engine import evaluate
from .env import write_environment
from .metrics import compute_metrics, per_class_table
from .models import build_model, display_name
from .utils import ensure_dir, get_device, load_config, resolve_data_dir, set_seed


def _history_stats(results_dir: str, model: str) -> dict:
    path = os.path.join(results_dir, f"history_{model}.csv")
    if not os.path.exists(path):
        return {}
    h = pd.read_csv(path)
    return {
        "epochs_run": int(h["epoch"].max() + 1),
        "mean_epoch_time_s": float(h["epoch_time_s"].mean()),
        "total_train_time_s": float(h["epoch_time_s"].sum()),
        "peak_vram_train_mb": float(h["peak_vram_mb"].max()),
        "total_train_energy_wh": float(h["gpu_energy_wh"].sum()) if h["gpu_energy_wh"].notna().any() else float("nan"),
        "best_val_f1_macro": float(h["val_f1_macro"].max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*", default=None, help="subconjunto; padrao = todos do config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg["deterministic"])
    device = get_device()
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    model_names = args.models or cfg["models"]

    write_environment(os.path.join(results_dir, "environment.json"),
                      cpu_threads=cfg["efficiency"]["cpu_threads"])

    data_dir = resolve_data_dir(cfg, args.data)
    loaders, class_weights, class_names, _ = build_dataloaders(cfg, data_dir)
    num_classes = len(class_names)
    criterion = nn.CrossEntropyLoss()

    perf_rows, eff_rows = [], []

    for name in model_names:
        best_ckpt = os.path.join(cfg["paths"]["checkpoints_dir"], name, "best.pt")

        # ---- metricas preditivas (precisa do checkpoint treinado) ----
        if os.path.exists(best_ckpt):
            print(f"\n[test] {display_name(name)}")
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            model = build_model(name, num_classes, pretrained=False).to(device)
            model.load_state_dict(ckpt["model"])
            _, y_true, y_pred, y_prob = evaluate(model, loaders["test"], criterion, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            np.savez_compressed(
                os.path.join(results_dir, f"test_predictions_{name}.npz"),
                y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                class_names=np.array(class_names),
            )
            m = compute_metrics(y_true, y_pred, y_prob, class_names)
            m = {"model": name, **m, **_history_stats(results_dir, name)}
            perf_rows.append(m)
            per_class_table(y_true, y_pred, class_names).to_csv(
                os.path.join(results_dir, f"per_class_{name}.csv"), index=False)
            print("   " + " | ".join(
                f"{k}={m[k]:.4f}" for k in ["accuracy", "balanced_accuracy", "f1_macro", "auc_macro_ovr"]))
        else:
            print(f"[test] sem checkpoint para {name} ({best_ckpt}); pulando metricas preditivas.")

        # ---- metricas de eficiencia (intrinsecas a arquitetura) ----
        print(f"[eff] perfilando {display_name(name)} ...")
        eff = profile_model(name, num_classes, device, cfg["efficiency"])
        eff.update({k: v for k, v in _history_stats(results_dir, name).items()
                    if k in ("mean_epoch_time_s", "total_train_time_s",
                             "peak_vram_train_mb", "total_train_energy_wh")})
        eff_rows.append(eff)

    perf_df = pd.DataFrame(perf_rows)
    eff_df = pd.DataFrame(eff_rows)
    if not perf_df.empty:
        perf_df.to_csv(os.path.join(results_dir, "performance.csv"), index=False)
    eff_df.to_csv(os.path.join(results_dir, "efficiency.csv"), index=False)

    # ---- resumo com custo-beneficio ----
    if not perf_df.empty:
        summary = perf_df[["model", "accuracy", "balanced_accuracy", "f1_macro",
                           "f1_weighted", "auc_macro_ovr"]].merge(
            eff_df[["model", "params_total_m", "size_mb", "gmacs",
                    "cpu_latency_ms_mean", "peak_vram_train_mb"]], on="model", how="outer")
        summary["f1_per_gmac"] = summary["f1_macro"] / summary["gmacs"]
        summary["f1_per_mparam"] = summary["f1_macro"] / summary["params_total_m"]
        summary["display"] = summary["model"].map(display_name)
        summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
        print("\n==== RESUMO ====")
        with pd.option_context("display.width", 160, "display.max_columns", None):
            print(summary.round(4).to_string(index=False))

    print(f"\n[benchmark] artefatos em {results_dir}/")


if __name__ == "__main__":
    main()
