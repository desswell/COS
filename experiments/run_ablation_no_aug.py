"""Ablation: OwnCNN without augmentation, 3 folds. Demonstrates aug effect."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We reuse run_own_cnn's run_one_fold + main scaffolding
from experiments.run_own_cnn import run_one_fold
from src.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--out-name", default="ablation_no_aug")
    args = parser.parse_args()
    # Force augmentation off
    args.augment = False
    args.fast = False

    results_dir = Path(RESULTS_DIR) / args.out_name
    results_dir.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        print(f"\n========== FOLD {fold} (no-aug ablation) ==========")
        run_one_fold(fold, args, results_dir)


if __name__ == "__main__":
    main()
