from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, float] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["auroc"] = float("nan")

    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        metrics["auprc"] = float("nan")

    return metrics


def optimize_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1",
) -> dict[str, float]:
    if metric not in {"f1", "balanced_accuracy", "youden_j"}:
        raise ValueError("metric must be one of: f1, balanced_accuracy, youden_j")

    candidates = np.unique(np.concatenate(([0.0], np.clip(y_score, 0.0, 1.0), [1.0])))
    best_thr = 0.5
    best_score = -1.0

    for thr in candidates:
        y_pred = (y_score >= thr).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_true, y_pred))
        else:
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tpr = tp / max(tp + fn, 1)
            tnr = tn / max(tn + fp, 1)
            score = float(tpr + tnr - 1.0)

        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return {
        "metric": metric,
        "threshold": float(best_thr),
        "score": float(best_score),
    }
