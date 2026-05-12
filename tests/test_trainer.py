import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer, TrainConfig


class _Tiny(nn.Module):
    def __init__(self, n_features=8, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


def _make_loaders():
    torch.manual_seed(0)
    n = 80
    x = torch.randn(n, 1, 4, 2)  # 1x4x2 = 8 features
    y = torch.randint(0, 3, (n,))
    train_ds = TensorDataset(x[:60], y[:60])
    val_ds = TensorDataset(x[60:], y[60:])
    return DataLoader(train_ds, batch_size=8, shuffle=True), DataLoader(val_ds, batch_size=8)


def test_trainer_fits_one_epoch_and_returns_history(tmp_path):
    model = _Tiny()
    train_dl, val_dl = _make_loaders()
    cfg = TrainConfig(
        epochs=2,
        lr=1e-2,
        weight_decay=0.0,
        early_stop_patience=10,
        grad_clip=1.0,
        num_classes=3,
        log_csv=tmp_path / "log.csv",
        ckpt_path=tmp_path / "best.pt",
    )
    trainer = Trainer(model, train_dl, val_dl, cfg)
    history = trainer.fit()

    assert "epochs" in history
    assert len(history["epochs"]) == 2
    assert "best_val_acc" in history
    assert (tmp_path / "log.csv").exists()
    assert (tmp_path / "best.pt").exists()


def test_trainer_early_stop_triggers(tmp_path):
    model = _Tiny()
    train_dl, val_dl = _make_loaders()
    cfg = TrainConfig(
        epochs=50,
        lr=1e-2,
        weight_decay=0.0,
        early_stop_patience=2,  # tight patience
        grad_clip=1.0,
        num_classes=3,
        log_csv=tmp_path / "log.csv",
        ckpt_path=tmp_path / "best.pt",
    )
    trainer = Trainer(model, train_dl, val_dl, cfg)
    history = trainer.fit()
    # We should NOT have run all 50 epochs on a 60-sample noise dataset
    assert len(history["epochs"]) < 50


def test_trainer_evaluate_returns_metrics_and_preds(tmp_path):
    model = _Tiny()
    _, val_dl = _make_loaders()
    cfg = TrainConfig(
        epochs=1, lr=1e-2, weight_decay=0.0, early_stop_patience=10,
        grad_clip=1.0, num_classes=3,
        log_csv=tmp_path / "log.csv", ckpt_path=tmp_path / "best.pt",
    )
    trainer = Trainer(model, val_dl, val_dl, cfg)
    metrics, y_true, y_pred = trainer.evaluate(val_dl)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert y_true.shape == y_pred.shape
    assert y_pred.dtype.kind in ("i", "u")
