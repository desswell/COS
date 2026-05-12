"""Generic training loop.

Agnostic to model architecture and dataset — just give it a `nn.Module`, two
DataLoaders, and a `TrainConfig`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.training.metrics import compute_metrics
from src.utils.io import append_csv_row


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    early_stop_patience: int
    grad_clip: float
    num_classes: int
    log_csv: str | Path
    ckpt_path: str | Path
    # Epoch numbering offset for CSV log (useful when training in phases on the same CSV).
    # Epochs written: epoch_offset+1 .. epoch_offset+epochs
    epoch_offset: int = 0
    # Optional: scheduler factory and optimizer factory for advanced use (e.g. PANN param groups)
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None
    scheduler_factory: Callable[[torch.optim.Optimizer, int], torch.optim.lr_scheduler.LRScheduler] | None = None


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()

        if cfg.optimizer_factory is not None:
            self.optimizer = cfg.optimizer_factory(model)
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )

        if cfg.scheduler_factory is not None:
            self.scheduler = cfg.scheduler_factory(self.optimizer, cfg.epochs)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs
            )

        Path(cfg.log_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        total_correct = 0
        total = 0
        for x, y in loader:
            x = x.float()
            y = y.long()
            with torch.set_grad_enabled(train):
                logits = self.model(x)
                loss = self.criterion(logits, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        return total_loss / max(total, 1), total_correct / max(total, 1)

    def fit(self) -> dict:
        best_val_acc = -1.0
        best_epoch = -1
        epochs_since_improve = 0
        history = {"epochs": [], "best_val_acc": 0.0, "best_epoch": 0, "total_time_s": 0.0}
        t_start = time.perf_counter()

        for local_epoch in range(1, self.cfg.epochs + 1):
            epoch = self.cfg.epoch_offset + local_epoch
            t0 = time.perf_counter()
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            dt = time.perf_counter() - t0

            # Also compute macro-F1 on val for logging
            _, y_true, y_pred = self.evaluate(self.val_loader, set_train_back=True)
            metrics = compute_metrics(y_true, y_pred, num_classes=self.cfg.num_classes)
            val_f1 = metrics["macro_f1"]

            append_csv_row(self.cfg.log_csv, {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "val_f1": round(val_f1, 6),
                "lr": round(lr, 8),
                "time_s": round(dt, 3),
            })
            history["epochs"].append(epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_since_improve = 0
                torch.save({"model": self.model.state_dict(), "epoch": epoch}, self.cfg.ckpt_path)
            else:
                epochs_since_improve += 1

            tqdm.write(
                f"epoch {epoch:3d} | train {train_loss:.4f}/{train_acc:.3f} "
                f"val {val_loss:.4f}/{val_acc:.3f} (best {best_val_acc:.3f}@{best_epoch}) "
                f"lr={lr:.2e} {dt:.1f}s"
            )

            if epochs_since_improve >= self.cfg.early_stop_patience:
                tqdm.write(f"  early stop at epoch {epoch} (patience {self.cfg.early_stop_patience})")
                break

        # If nothing ever improved past initial -1.0, save current weights so load_best() works
        if not Path(self.cfg.ckpt_path).exists():
            torch.save({"model": self.model.state_dict(), "epoch": -1}, self.cfg.ckpt_path)

        history["best_val_acc"] = best_val_acc
        history["best_epoch"] = best_epoch
        history["total_time_s"] = time.perf_counter() - t_start
        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, set_train_back: bool = False):
        was_training = self.model.training
        self.model.eval()
        all_true, all_pred = [], []
        for x, y in loader:
            x = x.float()
            logits = self.model(x)
            pred = logits.argmax(dim=1)
            all_true.append(y.numpy())
            all_pred.append(pred.cpu().numpy())
        y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
        y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)
        metrics = compute_metrics(y_true, y_pred, num_classes=self.cfg.num_classes)
        if set_train_back and was_training:
            self.model.train()
        return metrics, y_true, y_pred

    def load_best(self) -> None:
        ckpt = torch.load(self.cfg.ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
