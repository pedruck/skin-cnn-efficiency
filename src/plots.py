"""Gera todas as figuras do artigo a partir de ``results/`` (PDF vetorial + PNG 300 dpi).

Uso:
    python -m src.plots                       # figuras que dependem so de results/
    python -m src.plots --data /isic2019      # inclui a grade de amostras (Fig. 1)
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .models import display_name  # noqa: E402
from .utils import ensure_dir, load_config  # noqa: E402

# paleta segura para daltonismo (Okabe-Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _save(fig, figures_dir: str, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figures_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[plot] {name}")


def _models_with_predictions(results_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(results_dir, "test_predictions_*.npz")))
    return [os.path.basename(f)[len("test_predictions_"):-len(".npz")] for f in files]


# --------------------------------------------------------------------------- #
def fig_class_distribution(results_dir: str, figures_dir: str) -> None:
    path = os.path.join(results_dir, "splits.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    pivot = df.pivot_table(index="class", columns="split", values="image_base",
                           aggfunc="count", fill_value=0)
    pivot = pivot.reindex(columns=[c for c in ["train", "val", "test"] if c in pivot.columns])
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bottom = np.zeros(len(pivot))
    for i, split in enumerate(pivot.columns):
        ax.bar(pivot.index, pivot[split], bottom=bottom, label=split, color=PALETTE[i])
        bottom += pivot[split].to_numpy()
    ax.set_ylabel("Numero de imagens")
    ax.set_xlabel("Classe diagnostica")
    ax.set_title("Distribuicao de classes do ISIC 2019 (escala log)")
    ax.set_yscale("log")
    ax.legend(title="Particao", frameon=False)
    _save(fig, figures_dir, "fig2_class_distribution")


def fig_training_curves(results_dir: str, figures_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(results_dir, "history_*.csv")))
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for i, f in enumerate(files):
        name = os.path.basename(f)[len("history_"):-len(".csv")]
        h = pd.read_csv(f)
        ep = h["epoch"] + 1
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(ep, h["train_loss"], "--", color=c, alpha=0.7)
        axes[0].plot(ep, h["val_loss"], "-", color=c, label=display_name(name))
        axes[1].plot(ep, h["val_f1_macro"], "-", color=c, label=display_name(name))
    axes[0].set_title("Perda (tracejado: treino / solido: val)")
    axes[0].set_xlabel("Epoca"); axes[0].set_ylabel("Cross-entropy")
    axes[1].set_title("Macro-F1 de validacao")
    axes[1].set_xlabel("Epoca"); axes[1].set_ylabel("Macro-F1")
    axes[1].legend(frameon=False, fontsize=7)
    _save(fig, figures_dir, "fig3_training_curves")


def fig_tradeoff(results_dir: str, figures_dir: str) -> None:
    path = os.path.join(results_dir, "summary.csv")
    if not os.path.exists(path):
        return
    s = pd.read_csv(path).dropna(subset=["f1_macro"])
    if s.empty:
        return
    x_options = [
        ("gmacs", "Custo computacional (GMACs)"),
        ("cpu_latency_ms_mean", "Latencia em CPU (ms/imagem)"),
        ("params_total_m", "Parametros (milhoes)"),
    ]
    x_options = [(c, lbl) for c, lbl in x_options if c in s.columns and s[c].notna().any()]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(s))]
    fig, axes = plt.subplots(1, len(x_options), figsize=(3.4 * len(x_options), 3.2), squeeze=False)
    for ax, (col, lbl) in zip(axes[0], x_options):
        sizes = 40 + 240 * (s["params_total_m"] / s["params_total_m"].max())
        ax.scatter(s[col], s["f1_macro"], s=sizes, c=colors, alpha=0.85,
                   edgecolors="black", linewidths=0.5)
        for _, r in s.iterrows():
            ax.annotate(r["display"], (r[col], r["f1_macro"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set_xlabel(lbl); ax.set_ylabel("Macro-F1 (teste)")
    fig.suptitle("Compromisso desempenho x custo (marcador ~ nº de parametros)")
    _save(fig, figures_dir, "fig4_tradeoff")


def fig_confusion(results_dir: str, figures_dir: str) -> None:
    from .metrics import confusion

    for name in _models_with_predictions(results_dir):
        d = np.load(os.path.join(results_dir, f"test_predictions_{name}.npz"), allow_pickle=True)
        classes = [str(c) for c in d["class_names"]]
        cm = confusion(d["y_true"], d["y_pred"], classes, normalize="true")

        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
        ax.set_title(f"Matriz de confusao normalizada - {display_name(name)}")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.grid(False)
        _save(fig, figures_dir, f"fig5_confusion_{name}")


def fig_per_class_f1(results_dir: str, figures_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(results_dir, "per_class_*.csv")))
    if not files:
        return
    frames = {os.path.basename(f)[len("per_class_"):-len(".csv")]: pd.read_csv(f) for f in files}
    classes = next(iter(frames.values()))["class"].tolist()
    x = np.arange(len(classes))
    width = 0.8 / len(frames)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for i, (name, df) in enumerate(frames.items()):
        ax.bar(x + i * width, df["f1"], width, label=display_name(name), color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * (len(frames) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.set_ylabel("F1 por classe (teste)")
    ax.set_title("Desempenho por classe")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, figures_dir, "fig6_per_class_f1")


def fig_efficiency_bars(results_dir: str, figures_dir: str) -> None:
    path = os.path.join(results_dir, "efficiency.csv")
    if not os.path.exists(path):
        return
    e = pd.read_csv(path)
    e["display"] = e["model"].map(display_name)
    metrics = [
        ("params_total_m", "Parametros (M)"),
        ("gmacs", "GMACs"),
        ("cpu_latency_ms_mean", "Latencia CPU (ms)"),
        ("gpu_latency_ms_mean", "Latencia GPU (ms)"),
        ("size_mb", "Tamanho (MB)"),
        ("peak_vram_train_mb", "VRAM pico treino (MB)"),
    ]
    metrics = [(c, lbl) for c, lbl in metrics if c in e.columns and e[c].notna().any()]
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow), squeeze=False)
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(e))]
    for ax, (col, lbl) in zip(axes.flat, metrics):
        ax.bar(e["display"], e[col], color=bar_colors)
        ax.set_title(lbl)
        ax.tick_params(axis="x", rotation=30)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle("Metricas de custo de hardware")
    _save(fig, figures_dir, "fig7_efficiency_bars")


def fig_pr_curves(results_dir: str, figures_dir: str) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    names = _models_with_predictions(results_dir)
    if not names:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for i, name in enumerate(names):
        d = np.load(os.path.join(results_dir, f"test_predictions_{name}.npz"), allow_pickle=True)
        y_true, y_prob = d["y_true"], d["y_prob"]
        n = y_prob.shape[1]
        y_onehot = np.eye(n)[y_true]
        p, r, _ = precision_recall_curve(y_onehot.ravel(), y_prob.ravel())
        ap = average_precision_score(y_onehot, y_prob, average="micro")
        ax.plot(r, p, color=PALETTE[i % len(PALETTE)], label=f"{display_name(name)} (AP={ap:.3f})")
    ax.set_xlabel("Revocacao"); ax.set_ylabel("Precisao")
    ax.set_title("Curva precisao-revocacao (micro-media)")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, figures_dir, "fig8_pr_curves")


def fig_samples(data_dir: str, cfg: dict, figures_dir: str) -> None:
    from PIL import Image

    from .data import build_master_frame

    classes = list(cfg["data"]["classes"])
    frame = build_master_frame(data_dir, classes)
    fig, axes = plt.subplots(1, len(classes), figsize=(1.5 * len(classes), 1.8))
    rng = np.random.default_rng(cfg["seed"])
    for ax, cls in zip(axes, classes):
        sub = frame[frame["class"] == cls]
        if sub.empty:
            ax.set_visible(False)
            continue
        row = sub.iloc[int(rng.integers(len(sub)))]
        ax.imshow(Image.open(row["path"]).convert("RGB").resize((160, 160)))
        ax.set_title(cls, fontsize=9)
        ax.axis("off")
    fig.suptitle("Exemplos por classe - ISIC 2019")
    _save(fig, figures_dir, "fig1_samples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="diretorio do ISIC 2019 (para a Fig. 1)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg["paths"]["results_dir"]
    figures_dir = ensure_dir(cfg["paths"]["figures_dir"])

    if args.data:
        try:
            fig_samples(args.data, cfg, figures_dir)
        except Exception as exc:  # nao trava o resto das figuras
            print(f"[plot] fig1_samples pulada: {exc}")

    fig_class_distribution(results_dir, figures_dir)
    fig_training_curves(results_dir, figures_dir)
    fig_tradeoff(results_dir, figures_dir)
    fig_confusion(results_dir, figures_dir)
    fig_per_class_f1(results_dir, figures_dir)
    fig_efficiency_bars(results_dir, figures_dir)
    fig_pr_curves(results_dir, figures_dir)
    print(f"\n[plots] figuras em {figures_dir}/")


if __name__ == "__main__":
    main()
