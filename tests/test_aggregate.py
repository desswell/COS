from pathlib import Path

import numpy as np
import pandas as pd

from experiments.aggregate_results import (
    aggregate_one_model, build_summary_md, average_confusion_matrices,
)


def _write_folds_summary(dir_: Path, accs, f1s, params=250_000):
    rows = []
    for fold, (a, f) in enumerate(zip(accs, f1s), 1):
        rows.append({
            "fold": fold,
            "accuracy": a,
            "macro_f1": f,
            "best_epoch": 10 + fold,
            "train_time_s": 600,
            "inference_time_ms_per_sample": 5.0,
            "n_params": params,
        })
    dir_.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dir_ / "folds_summary.csv", index=False)
    # Also write fake confusion matrices
    for r in rows:
        cm = np.eye(10, dtype=int) * 80
        np.save(dir_ / f"fold_{r['fold']}_cm.npy", cm)


def test_aggregate_one_model_mean_std(tmp_path):
    d = tmp_path / "own_cnn"
    _write_folds_summary(d, accs=[0.75, 0.80, 0.85], f1s=[0.70, 0.78, 0.82])
    summary = aggregate_one_model("own_cnn", d)
    assert summary["model"] == "own_cnn"
    assert summary["n_folds"] == 3
    assert abs(summary["acc_mean"] - 0.80) < 1e-6
    assert summary["acc_std"] > 0


def test_average_confusion_matrices(tmp_path):
    d = tmp_path / "m"
    _write_folds_summary(d, accs=[0.9, 0.9], f1s=[0.9, 0.9])
    avg = average_confusion_matrices(d, n_classes=10)
    assert avg.shape == (10, 10)
    # Diagonal should dominate
    assert (np.diag(avg) > 0).all()


def test_build_summary_md_contains_each_row(tmp_path):
    rows = [
        {"model": "own_cnn", "n_folds": 10, "acc_mean": 0.78, "acc_std": 0.03,
         "f1_mean": 0.77, "f1_std": 0.04, "train_time_s_total": 7200,
         "n_params": 250000, "best_epoch_avg": 25},
        {"model": "pann_cnn10", "n_folds": 10, "acc_mean": 0.89, "acc_std": 0.02,
         "f1_mean": 0.88, "f1_std": 0.02, "train_time_s_total": 14400,
         "n_params": 5_000_000, "best_epoch_avg": 15},
    ]
    md = build_summary_md(rows)
    assert "own_cnn" in md
    assert "pann_cnn10" in md
    assert "0.78" in md and "0.89" in md
