"""Metricas preditivas apropriadas a um dataset desbalanceado (ISIC 2019)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                    class_names: list[str], clinical_class: str = "MEL") -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(class_names)
    labels = list(range(n))

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
    }

    try:
        out["auc_macro_ovr"] = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
    except Exception:
        out["auc_macro_ovr"] = float("nan")

    if clinical_class in class_names:
        ci = class_names.index(clinical_class)
        rec = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0)[1]
        out[f"recall_{clinical_class}"] = float(rec[ci])

    return out


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({
        "class": class_names,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    })


def confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str],
              normalize: str | None = "true") -> np.ndarray:
    labels = list(range(len(class_names)))
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)


# --------------------------------------------------------------------------- #
# Intervalos de confianca por bootstrap
# --------------------------------------------------------------------------- #
def _macro_f1_fast(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 identico ao de ``f1_score(average='macro', labels=todas,
    zero_division=0)``, mas sem o overhead de chamada do sklearn -- o bootstrap
    executa isto milhares de vezes.

    A matriz de confusao sai de um unico ``bincount`` sobre ``y_true*K + y_pred``.
    Classe ausente de y_true e de y_pred entra no macro com F1 = 0, como no sklearn.
    """
    cm = np.bincount(y_true * n_classes + y_pred,
                     minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(f1.mean())


def bootstrap_f1_ci(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str],
                    n_resamples: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Intervalo percentil para o macro-F1, reamostrando o teste com reposicao.

    Responde: quanto o resultado mudaria se outras N imagens tivessem sido
    sorteadas para o conjunto de teste? Nao captura variacao de semente de
    treino -- e um piso da incerteza total, nao o total.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    k, n = len(class_names), len(y_true)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        scores[b] = _macro_f1_fast(y_true[idx], y_pred[idx], k)
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "f1_macro_ci_low": float(lo),
        "f1_macro_ci_high": float(hi),
        "f1_macro_boot_se": float(scores.std(ddof=1)),
    }


def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     class_names: list[str], n_resamples: int = 1000,
                     seed: int = 42, alpha: float = 0.05) -> dict:
    """Compara dois modelos reamostrando os MESMOS indices para ambos.

    Como as duas arquiteturas foram avaliadas nas mesmas imagens, o pareamento
    remove a variabilidade de "quais imagens sairam no sorteio" e isola a
    diferenca entre modelos. E bem mais sensivel do que verificar se dois
    intervalos calculados em separado se sobrepoem.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)
    k, n = len(class_names), len(y_true)

    observed = _macro_f1_fast(y_true, pred_a, k) - _macro_f1_fast(y_true, pred_b, k)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        diffs[b] = _macro_f1_fast(yt, pred_a[idx], k) - _macro_f1_fast(yt, pred_b[idx], k)

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # p bilateral do bootstrap: o dobro da massa do lado oposto ao efeito observado
    p = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {
        "delta_f1_macro": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "significant": bool(lo > 0 or hi < 0),
    }
