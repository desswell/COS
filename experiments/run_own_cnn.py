"""10-fold CV training of OwnCNN on UrbanSound8K log-mel spectrograms."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.config import (
    OWN_AUDIO, NUM_CLASSES, OWN_CNN_TRAIN,
    METADATA_PATH, AUDIO_SUBDIR, CACHE_DIR_OWN, DATA_DIR, RESULTS_DIR, CLASS_NAMES,
)
from src.data.dataset import UrbanSound8KDataset
from src.models.own_cnn import OwnCNN
from src.training.seed import seed_everything
from src.training.trainer import Trainer, TrainConfig
from src.utils.io import append_csv_row, save_npy, write_json


def split_train_val(dataset, val_frac: float, seed: int):
    """Random 90/10 split of a Dataset, with fixed seed."""
    n = len(dataset)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return Subset(dataset, tr_idx.tolist()), Subset(dataset, val_idx.tolist())


def run_one_fold(test_fold: int, args, results_dir: Path) -> dict:
    seed_everything(test_fold * 100 + 42)

    train_folds = [f for f in range(1, 11) if f != test_fold]
    full_train = UrbanSound8KDataset(
        metadata_csv=METADATA_PATH,
        audio_root=Path(DATA_DIR) / AUDIO_SUBDIR,
        folds=train_folds,
        train=True,
        augment=args.augment,
        cache_dir=CACHE_DIR_OWN,
        audio_config=OWN_AUDIO,
    )
    test_ds = UrbanSound8KDataset(
        metadata_csv=METADATA_PATH,
        audio_root=Path(DATA_DIR) / AUDIO_SUBDIR,
        folds=[test_fold],
        train=False,
        augment=False,
        cache_dir=CACHE_DIR_OWN,
        audio_config=OWN_AUDIO,
    )
    tr_ds, val_ds = split_train_val(full_train, val_frac=0.10, seed=42)

    bs = OWN_CNN_TRAIN["batch_size"]
    train_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    model = OwnCNN(num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=OWN_CNN_TRAIN["lr"],
        weight_decay=OWN_CNN_TRAIN["weight_decay"],
        early_stop_patience=OWN_CNN_TRAIN["early_stop_patience"],
        grad_clip=OWN_CNN_TRAIN["grad_clip"],
        num_classes=NUM_CLASSES,
        log_csv=results_dir / f"fold_{test_fold}.csv",
        ckpt_path=results_dir / f"fold_{test_fold}_best.pt",
    )
    trainer = Trainer(model, train_dl, val_dl, cfg)
    history = trainer.fit()
    trainer.load_best()

    import time
    n_test = len(test_ds)
    t0 = time.perf_counter()
    metrics, y_true, y_pred = trainer.evaluate(test_dl)
    inf_time_ms = (time.perf_counter() - t0) * 1000.0 / max(n_test, 1)

    save_npy(results_dir / f"fold_{test_fold}_cm.npy", metrics["confusion_matrix"])
    write_json(results_dir / f"fold_{test_fold}_per_class.json", {
        "f1": metrics["per_class_f1"].tolist(),
        "precision": metrics["per_class_precision"].tolist(),
        "recall": metrics["per_class_recall"].tolist(),
        "class_names": list(CLASS_NAMES),
    })

    summary_row = {
        "fold": test_fold,
        "accuracy": round(float(metrics["accuracy"]), 6),
        "macro_f1": round(float(metrics["macro_f1"]), 6),
        "best_epoch": history["best_epoch"],
        "train_time_s": round(history["total_time_s"], 1),
        "inference_time_ms_per_sample": round(inf_time_ms, 3),
        "n_params": n_params,
    }
    append_csv_row(results_dir / "folds_summary.csv", summary_row)
    return summary_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--epochs", type=int, default=OWN_CNN_TRAIN["epochs"])
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--fast", action="store_true", help="1 fold, 2 epochs, no augment")
    parser.add_argument("--out-name", default="own_cnn", help="subdir of results/")
    args = parser.parse_args()

    if args.fast:
        args.folds = [1]
        args.epochs = 2

    results_dir = Path(RESULTS_DIR) / args.out_name
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in args.folds:
        print(f"\n========== FOLD {fold} ==========")
        row = run_one_fold(fold, args, results_dir)
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    print("\n=== Summary across folds ===")
    print(df.describe())
    df.to_csv(results_dir / "folds_summary.csv", index=False)


if __name__ == "__main__":
    main()
