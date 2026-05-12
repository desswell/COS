"""Classification metrics: accuracy, macro F1, per-class precision/recall/F1, confusion matrix."""
from __future__ import annotations

from typing import TypedDict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricsDict(TypedDict):
    accuracy: float
    macro_f1: float
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    confusion_matrix: np.ndarray


def compute_metrics(y_true, y_pred, num_classes: int) -> MetricsDict:
    """Compute classification metrics from 1D arrays of integer labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_precision": precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ),
        "per_class_recall": recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ),
        "per_class_f1": f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }
