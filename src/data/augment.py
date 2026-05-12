"""Spectrogram-domain augmentations applied inside the Dataset."""
from __future__ import annotations

import torch
from torch import nn


class SpecAugment(nn.Module):
    """SpecAugment (Park et al., 2019) — time and frequency masking.

    Masked regions are set to the spectrogram's mean (a soft "blanking" that
    keeps the model from anchoring on the masked-region statistics).
    """

    def __init__(
        self,
        time_mask_param: int = 25,
        freq_mask_param: int = 12,
        n_time: int = 2,
        n_freq: int = 2,
    ) -> None:
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.n_time = n_time
        self.n_freq = n_freq

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [C, F, T]
        assert spec.dim() == 3, f"expected [C,F,T], got {spec.shape}"
        out = spec.clone()
        fill = out.mean().item()
        c, f, t = out.shape

        for _ in range(self.n_freq):
            w = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
            if w == 0 or w >= f:
                continue
            start = int(torch.randint(0, f - w, (1,)).item())
            out[:, start : start + w, :] = fill

        for _ in range(self.n_time):
            w = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
            if w == 0 or w >= t:
                continue
            start = int(torch.randint(0, t - w, (1,)).item())
            out[:, :, start : start + w] = fill

        return out


class GaussianNoise(nn.Module):
    """Add zero-mean Gaussian noise to the spectrogram with probability p."""

    def __init__(self, std: float = 0.005, p: float = 0.3) -> None:
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return spec
        if torch.rand(1).item() > self.p:
            return spec
        return spec + torch.randn_like(spec) * self.std
