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
