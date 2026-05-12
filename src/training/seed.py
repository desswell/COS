"""Deterministic seeding for reproducibility."""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python random, NumPy, PyTorch (CPU+CUDA) and PYTHONHASHSEED.

    Note: PyTorch operations on CUDA may still be non-deterministic unless
    `torch.use_deterministic_algorithms(True)` is set; we skip that since we
    run CPU-only.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
