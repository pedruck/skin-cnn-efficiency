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
