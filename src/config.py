"""Centralised constants for preprocessing and training.

All experiment scripts and dataset code MUST import from here — do not
hard-code these values anywhere else.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    duration_s: float
    n_fft: int
    hop_length: int
    n_mels: int
    f_min: float
    f_max: float

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.duration_s)


# Own CNN: 22050 Hz, 4 s, 64-band mel
OWN_AUDIO = AudioConfig(
    sample_rate=22050,
    duration_s=4.0,
    n_fft=1024,
    hop_length=512,
    n_mels=64,
    f_min=20.0,
    f_max=11025.0,
)

# PANN CNN10: 32 kHz, 4 s, 64-band mel matching PANN training settings
PANN_AUDIO = AudioConfig(
    sample_rate=32000,
    duration_s=4.0,
    n_fft=1024,
    hop_length=320,
    n_mels=64,
    f_min=50.0,
    f_max=14000.0,
)

NUM_CLASSES = 10
CLASS_NAMES = (
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
)

# Training defaults
OWN_CNN_TRAIN = dict(
    epochs=50,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    early_stop_patience=10,
    grad_clip=1.0,
)
PANN_TRAIN = dict(
    epochs=30,
    batch_size=16,
    lr_head=1e-3,
    lr_backbone=1e-4,
    weight_decay=1e-4,
    warmup_epochs=5,
    early_stop_patience=8,
    grad_clip=1.0,
)

# Paths (relative to project root)
DATA_DIR = "data/UrbanSound8K"
AUDIO_SUBDIR = "audio"
METADATA_PATH = "data/UrbanSound8K/metadata/UrbanSound8K.csv"
CACHE_DIR_OWN = "data/UrbanSound8K/cache_mel"
CACHE_DIR_PANN = "data/UrbanSound8K/cache_mel_pann"
RESULTS_DIR = "results"
