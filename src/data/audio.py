"""Pure audio preprocessing — no model, no dataset.

These functions are deliberately stateless and easy to unit-test.
"""
from pathlib import Path

import torch
import torchaudio

from src.config import AudioConfig


def load_audio(path: str | Path, target_sr: int) -> torch.Tensor:
    """Load an audio file, downmix to mono and resample to target_sr.

    Returns a 1D float32 tensor.
    """
    waveform, sr = torchaudio.load(str(path))  # [C, T] float32
    # Downmix to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).contiguous()


def pad_or_crop(wav: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Pad with zeros on the right (short) or center-crop (long) to exactly n_samples."""
    assert wav.dim() == 1, f"expected 1D waveform, got {wav.shape}"
    cur = wav.shape[0]
    if cur == n_samples:
        return wav
    if cur < n_samples:
        pad = torch.zeros(n_samples - cur, dtype=wav.dtype)
        return torch.cat([wav, pad], dim=0)
    # cur > n_samples — center crop
    start = (cur - n_samples) // 2
    return wav[start : start + n_samples].contiguous()


def log_mel_spectrogram(wav: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    """Compute log-mel spectrogram. Returns [1, n_mels, T] float32 tensor."""
    assert wav.dim() == 1, f"expected 1D waveform, got {wav.shape}"
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        power=2.0,
        center=True,
    )
    mel = mel_transform(wav)  # [n_mels, T]
    log_mel = torch.log10(mel + 1e-10)
    return log_mel.unsqueeze(0).to(torch.float32)


def normalize(spec: torch.Tensor) -> torch.Tensor:
    """Per-sample z-score on a [C, M, T] spectrogram. Safe for constant input."""
    mean = spec.mean()
    std = spec.std()
    return (spec - mean) / (std + 1e-6)
