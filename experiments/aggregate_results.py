"""Aggregate per-fold CSVs into summary.csv + summary.md + figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import NUM_CLASSES, CLASS_NAMES, RESULTS_DIR
from src.utils.plotting import (
    plot_learning_curves, plot_confusion_matrix, plot_per_class_f1_comparison,
)


def aggregate_one_model(model_name: str, model_dir: Path) -> dict:
    df = pd.read_csv(model_dir / "folds_summary.csv")
    n_folds = len(df)
    return {
        "model": model_name,
        "n_folds": n_folds,
        "acc_mean": float(df["accuracy"].mean()),
        "acc_std": float(df["accuracy"].std(ddof=1) if n_folds > 1 else 0.0),
        "f1_mean": float(df["macro_f1"].mean()),
        "f1_std": float(df["macro_f1"].std(ddof=1) if n_folds > 1 else 0.0),
        "train_time_s_total": float(df["train_time_s"].sum()),
        "n_params": int(df["n_params"].iloc[0]),
        "best_epoch_avg": float(df["best_epoch"].mean()),
    }


def average_confusion_matrices(model_dir: Path, n_classes: int) -> np.ndarray:
    cms = sorted(model_dir.glob("fold_*_cm.npy"))
    if not cms:
        return np.zeros((n_classes, n_classes), dtype=float)
    arr = np.stack([np.load(p) for p in cms], axis=0).astype(float)
    return arr.mean(axis=0)


def build_summary_md(rows: list[dict]) -> str:
    header = (
        "| model | folds | acc (mean±std) | macro-F1 (mean±std) | params | "
        "avg best epoch | total train time (s) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        body += (
            f"| {r['model']} | {r['n_folds']} | "
            f"{r['acc_mean']:.3f}±{r['acc_std']:.3f} | "
            f"{r['f1_mean']:.3f}±{r['f1_std']:.3f} | "
            f"{r['n_params']:,} | "
            f"{r['best_epoch_avg']:.1f} | "
            f"{r['train_time_s_total']:.0f} |\n"
        )
    return header + body


def _per_class_f1_from_folds(model_dir: Path) -> np.ndarray:
    """Read per-fold per_class.json files and average the F1 vectors."""
    import json
    f1s = []
    for p in sorted(model_dir.glob("fold_*_per_class.json")):
        with p.open() as f:
            d = json.load(f)
        f1s.append(np.array(d["f1"]))
    if not f1s:
        return np.zeros(NUM_CLASSES)
    return np.stack(f1s, axis=0).mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=RESULTS_DIR)
    parser.add_argument(
        "--models", nargs="+",
        default=["own_cnn", "pann_cnn10", "ablation_no_aug"],
        help="subdirs of results/ to aggregate",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_class_f1_by_model = {}

    for m in args.models:
        d = root / m
        if not (d / "folds_summary.csv").exists():
            print(f"[warn] {d}/folds_summary.csv missing — skipping")
            continue
        rows.append(aggregate_one_model(m, d))

        # Confusion matrix figure
        cm = average_confusion_matrices(d, n_classes=NUM_CLASSES)
        plot_confusion_matrix(
            cm, class_names=list(CLASS_NAMES),
            out_png=figures_dir / f"confusion_matrix_{m}_avg.png",
            title=f"{m}: avg normalised confusion (10-fold)",
            normalize=True,
        )

        # Learning curves for fold 0 (i.e. fold 1, our first fold)
        # Prefer fold 1 — that's what every script uses as its first fold
        for first_fold in (1, 5, 10):
            log = d / f"fold_{first_fold}.csv"
            if log.exists():
                plot_learning_curves(
                    log, out_png=figures_dir / f"learning_curves_{m}_fold{first_fold}.png",
                    title=f"{m} — fold {first_fold}",
                )
                break

        per_class_f1_by_model[m] = _per_class_f1_from_folds(d)

    summary_csv = root / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"[ok] wrote {summary_csv}")

    md = build_summary_md(rows)
    (root / "summary.md").write_text(md, encoding="utf-8")
    print(f"[ok] wrote {root / 'summary.md'}")

    if len(per_class_f1_by_model) >= 2:
        plot_per_class_f1_comparison(
            per_class_f1_by_model, class_names=list(CLASS_NAMES),
            out_png=figures_dir / "per_class_f1_comparison.png",
        )
        print(f"[ok] wrote per-class F1 comparison figure")


if __name__ == "__main__":
    main()
