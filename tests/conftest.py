"""Shared pytest fixtures."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sine_wav_path(tmp_path) -> Path:
    """Generate a 2-second 440 Hz sine wave at 16 kHz mono."""
    sr = 16000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "sine.wav"
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def stereo_wav_path(tmp_path) -> Path:
    """Generate a 1-second stereo wav at 8 kHz."""
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    left = 0.3 * np.sin(2 * np.pi * 300 * t)
    right = 0.3 * np.sin(2 * np.pi * 600 * t)
    audio = np.stack([left, right], axis=1).astype(np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(path, audio, sr)
    return path


import csv


@pytest.fixture
def fake_us8k(tmp_path) -> dict:
    """Create a minimal UrbanSound8K-shaped tree with 4 wavs across 2 folds, 2 classes."""
    sr = 22050
    root = tmp_path / "UrbanSound8K"
    audio_root = root / "audio"
    (root / "metadata").mkdir(parents=True)
    (audio_root / "fold1").mkdir(parents=True)
    (audio_root / "fold2").mkdir(parents=True)

    files = [
        ("a.wav", "fold1", 0, "class_a"),
        ("b.wav", "fold1", 1, "class_b"),
        ("c.wav", "fold2", 0, "class_a"),
        ("d.wav", "fold2", 1, "class_b"),
    ]
    for name, fold, _, _ in files:
        t = np.linspace(0, 1, sr, endpoint=False)
        # Use different frequency per file so they're distinguishable
        freq = 200 + 100 * (sum(map(ord, name)) % 5)
        audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(audio_root / fold / name, audio, sr)

    metadata_csv = root / "metadata" / "UrbanSound8K.csv"
    with metadata_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"])
        for name, fold, cls_id, cls in files:
            fold_num = int(fold.replace("fold", ""))
            w.writerow([name, 0, 0.0, 1.0, 1, fold_num, cls_id, cls])

    return {"root": root, "audio_root": audio_root, "metadata_csv": metadata_csv}
