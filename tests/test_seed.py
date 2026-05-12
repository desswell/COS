import random
import numpy as np
import torch

from src.training.seed import seed_everything


def test_seed_everything_reproduces_python_random():
    seed_everything(123)
    a = [random.random() for _ in range(5)]
    seed_everything(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_reproduces_numpy():
    seed_everything(42)
    a = np.random.rand(10)
    seed_everything(42)
    b = np.random.rand(10)
    assert np.allclose(a, b)


def test_seed_everything_reproduces_torch():
    seed_everything(7)
    a = torch.rand(10)
    seed_everything(7)
    b = torch.rand(10)
    assert torch.allclose(a, b)


def test_seed_everything_different_seed_different_output():
    seed_everything(1)
    a = torch.rand(10)
    seed_everything(2)
    b = torch.rand(10)
    assert not torch.allclose(a, b)
