import numpy as np

from src.training.metrics import compute_metrics


def test_compute_metrics_perfect_classifier():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    out = compute_metrics(y_true, y_pred, num_classes=3)
    assert out["accuracy"] == 1.0
    assert abs(out["macro_f1"] - 1.0) < 1e-6
    assert out["confusion_matrix"].shape == (3, 3)
    assert np.array_equal(out["confusion_matrix"], np.diag([2, 2, 2]))


def test_compute_metrics_all_wrong():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    out = compute_metrics(y_true, y_pred, num_classes=2)
    assert out["accuracy"] == 0.0


def test_compute_metrics_per_class_f1_present():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    out = compute_metrics(y_true, y_pred, num_classes=3)
    assert "per_class_f1" in out
    assert out["per_class_f1"].shape == (3,)
    # class 1 has perfect recall (both true 1s predicted 1) but precision 2/3
    assert out["per_class_f1"][1] > 0


def test_compute_metrics_handles_class_with_no_predictions():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    # Class 1 never appears — should not crash
    out = compute_metrics(y_true, y_pred, num_classes=2)
    assert out["accuracy"] == 1.0
