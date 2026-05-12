"""UrbanSound8K Dataset with per-file spectrogram caching."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import AudioConfig
from src.data.audio import load_audio, pad_or_crop, log_mel_spectrogram, normalize
from src.data.augment import SpecAugment, GaussianNoise


class UrbanSound8KDataset(Dataset):
    """Loads UrbanSound8K files filtered by fold. Caches log-mel spectrograms.

    Output: (spectrogram [1, n_mels, T], label int).
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_root: str | Path,
        folds: Sequence[int],
        train: bool,
        augment: bool,
        cache_dir: str | Path,
        audio_config: AudioConfig,
    ) -> None:
        super().__init__()
        self.audio_root = Path(audio_root)
        self.cache_dir = Path(cache_dir)
        self.train = train
        self.augment = augment
        self.cfg = audio_config

        df = pd.read_csv(metadata_csv)
        df = df[df["fold"].isin(list(folds))].reset_index(drop=True)
        self.items = df[["slice_file_name", "fold", "classID"]].to_records(index=False)

        if augment:
            self.spec_aug = SpecAugment(
                time_mask_param=25, freq_mask_param=12, n_time=2, n_freq=2
            )
            self.noise_aug = GaussianNoise(std=0.005, p=0.3)
        else:
            self.spec_aug = None
            self.noise_aug = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        slice_name, fold, label = self.items[idx]
        cache_path = self.cache_dir / f"fold{fold}" / f"{slice_name}.pt"

        if cache_path.exists():
            spec = torch.load(cache_path)
        else:
            wav_path = self.audio_root / f"fold{fold}" / slice_name
            wav = load_audio(wav_path, target_sr=self.cfg.sample_rate)
            wav = pad_or_crop(wav, n_samples=self.cfg.n_samples)
            spec = log_mel_spectrogram(wav, self.cfg)
            spec = normalize(spec)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(spec, cache_path)

        if self.augment:
            spec = self.spec_aug(spec)
            spec = self.noise_aug(spec)

        return spec, int(label)
