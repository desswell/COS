"""10-fold CV fine-tune of PANN CNN10 on UrbanSound8K."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.config import (
    PANN_AUDIO, NUM_CLASSES, PANN_TRAIN,
    METADATA_PATH, AUDIO_SUBDIR, CACHE_DIR_PANN, DATA_DIR, RESULTS_DIR, CLASS_NAMES,
)
from src.data.dataset import UrbanSound8KDataset
from src.models.pann_cnn10 import PannCnn10Wrapper
from src.training.seed import seed_everything
from src.training.trainer import Trainer, TrainConfig
from src.utils.io import append_csv_row, save_npy, write_json


def split_train_val(dataset, val_frac: float, seed: int):
    n = len(dataset)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    return Subset(dataset, idx[n_val:].tolist()), Subset(dataset, idx[:n_val].tolist())


def run_one_fold(test_fold: int, args, results_dir: Path) -> dict:
    seed_everything(test_fold * 100 + 42)

    train_folds = [f for f in range(1, 11) if f != test_fold]
    full_train = UrbanSound8KDataset(
        metadata_csv=METADATA_PATH,
        audio_root=Path(DATA_DIR) / AUDIO_SUBDIR,
        folds=train_folds,
        train=True,
        augment=args.augment,
        cache_dir=CACHE_DIR_PANN,
        audio_config=PANN_AUDIO,
    )
    test_ds = UrbanSound8KDataset(
        metadata_csv=METADATA_PATH,
        audio_root=Path(DATA_DIR) / AUDIO_SUBDIR,
        folds=[test_fold],
        train=False,
        augment=False,
        cache_dir=CACHE_DIR_PANN,
        audio_config=PANN_AUDIO,
    )
    tr_ds, val_ds = split_train_val(full_train, val_frac=0.10, seed=42)
    bs = PANN_TRAIN["batch_size"]
    train_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    model = PannCnn10Wrapper(num_classes=NUM_CLASSES, pretrained=not args.no_pretrained)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Phase 1: frozen backbone, head only. Saves a temp ckpt that's discarded after phase 2.
    model.freeze_backbone()
    cfg_phase1 = TrainConfig(
        epochs=args.warmup_epochs,
        lr=PANN_TRAIN["lr_head"],
        weight_decay=PANN_TRAIN["weight_decay"],
        early_stop_patience=args.warmup_epochs + 1,  # don't early-stop phase 1
        grad_clip=PANN_TRAIN["grad_clip"],
        num_classes=NUM_CLASSES,
        log_csv=results_dir / f"fold_{test_fold}.csv",
        ckpt_path=results_dir / f"fold_{test_fold}_phase1.pt",
        epoch_offset=0,
    )
    print(f"[PANN] phase 1 (warmup, frozen) for {args.warmup_epochs} epochs")
    trainer = Trainer(model, train_dl, val_dl, cfg_phase1)
    trainer.fit()

    # Phase 2: unfreeze, different LRs per param group
    model.unfreeze_backbone()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def opt_factory(m):
        return torch.optim.AdamW(
            m.param_groups(lr_backbone=PANN_TRAIN["lr_backbone"], lr_head=PANN_TRAIN["lr_head"]),
            weight_decay=PANN_TRAIN["weight_decay"],
        )

    cfg_phase2 = TrainConfig(
        epochs=args.epochs - args.warmup_epochs,
        lr=PANN_TRAIN["lr_backbone"],  # used only if optimizer_factory not given
        weight_decay=PANN_TRAIN["weight_decay"],
        early_stop_patience=PANN_TRAIN["early_stop_patience"],
        grad_clip=PANN_TRAIN["grad_clip"],
        num_classes=NUM_CLASSES,
        log_csv=results_dir / f"fold_{test_fold}.csv",  # append to same CSV
        ckpt_path=results_dir / f"fold_{test_fold}_best.pt",
        epoch_offset=args.warmup_epochs,  # continue epoch numbering after phase 1
        optimizer_factory=opt_factory,
    )
    print(f"[PANN] phase 2 (unfreeze) for {args.epochs - args.warmup_epochs} epochs")
    trainer2 = Trainer(model, train_dl, val_dl, cfg_phase2)
    history = trainer2.fit()
    trainer2.load_best()

    # Discard phase-1 checkpoint
    (results_dir / f"fold_{test_fold}_phase1.pt").unlink(missing_ok=True)

    n_test = len(test_ds)
    t0 = time.perf_counter()
    metrics, y_true, y_pred = trainer2.evaluate(test_dl)
    inf_time_ms = (time.perf_counter() - t0) * 1000.0 / max(n_test, 1)

    save_npy(results_dir / f"fold_{test_fold}_cm.npy", metrics["confusion_matrix"])
    write_json(results_dir / f"fold_{test_fold}_per_class.json", {
        "f1": metrics["per_class_f1"].tolist(),
        "precision": metrics["per_class_precision"].tolist(),
        "recall": metrics["per_class_recall"].tolist(),
        "class_names": list(CLASS_NAMES),
    })

    row = {
        "fold": test_fold,
        "accuracy": round(float(metrics["accuracy"]), 6),
        "macro_f1": round(float(metrics["macro_f1"]), 6),
        "best_epoch": history["best_epoch"],
        "train_time_s": round(history["total_time_s"], 1),
        "inference_time_ms_per_sample": round(inf_time_ms, 3),
        "n_params": n_params,
    }
    append_csv_row(results_dir / "folds_summary.csv", row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--epochs", type=int, default=PANN_TRAIN["epochs"])
    parser.add_argument("--warmup-epochs", type=int, default=PANN_TRAIN["warmup_epochs"])
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip weight download (for smoke testing only)")
    parser.add_argument("--fast", action="store_true",
                        help="1 fold, warmup=1 epoch + 1 phase-2 epoch")
    parser.add_argument("--out-name", default="pann_cnn10")
    args = parser.parse_args()

    if args.fast:
        args.folds = [1]
        args.epochs = 2
        args.warmup_epochs = 1

    results_dir = Path(RESULTS_DIR) / args.out_name
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in args.folds:
        print(f"\n========== FOLD {fold} ==========")
        row = run_one_fold(fold, args, results_dir)
        rows.append(row)
        print(row)

    pd.DataFrame(rows).to_csv(results_dir / "folds_summary.csv", index=False)


if __name__ == "__main__":
    main()
