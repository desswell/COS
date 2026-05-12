"""Matplotlib helpers for learning curves, confusion matrices, and bar comparisons."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_learning_curves(csv_path: str | Path, out_png: str | Path, title: str = "") -> None:
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"], label="train acc")
    axes[1].plot(df["epoch"], df["val_acc"], label="val acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_png: str | Path,
    title: str = "",
    normalize: bool = True,
) -> None:
    cm = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_per_class_f1_comparison(
    f1_by_model: dict[str, np.ndarray],
    class_names: Sequence[str],
    out_png: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    n_models = len(f1_by_model)
    x = np.arange(len(class_names))
    width = 0.8 / n_models
    for i, (name, f1) in enumerate(f1_by_model.items()):
        ax.bar(x + i * width - 0.4 + width / 2, f1, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_aug_effect(original, after_specaug, after_noise, out_png: str | Path) -> None:
    """Plot a spectrogram before / after SpecAugment / after noise side-by-side."""
    specs = [(original, "Original"), (after_specaug, "+ SpecAugment"), (after_noise, "+ Noise")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (spec, title) in zip(axes, specs):
        s = spec.squeeze().numpy() if hasattr(spec, "numpy") else np.asarray(spec).squeeze()
        ax.imshow(s, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("frames")
        ax.set_ylabel("mel bins")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
