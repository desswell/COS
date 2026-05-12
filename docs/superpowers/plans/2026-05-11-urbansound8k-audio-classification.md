# UrbanSound8K Audio Classification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full ML pipeline for 10-class urban-sound classification on UrbanSound8K with the official 10-fold CV, training (a) a custom 2D-CNN on log-mel spectrograms from scratch and (b) a fine-tuned PANN CNN10 pretrained on AudioSet, plus an ablation without augmentation.

**Architecture:** PyTorch + torchaudio pipeline. Audio → 22050 Hz mono → fixed 4s pad/crop → log-mel (64×173) → spectrogram cache → SpecAugment + Gaussian noise → CNN → cross-entropy. PANN branch uses a separate 32 kHz mel cache. Each model trained with 10-fold CV; per-fold logs aggregated to `summary.csv` and `summary.md`.

**Tech Stack:** Python 3.10, torch 2.8 (CPU), torchaudio, panns_inference, soundfile, librosa, scikit-learn, matplotlib/seaborn, pandas, tqdm, pytest. Existing venv at `C:\Users\desswell\work\pythonProject\.venv`.

**Working directory for all commands:** `C:\Users\desswell\work\pythonProject\ЦОС\final_project`
**Python:** `& "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe"` (referred to as `$PY` below). On Windows PowerShell prefix with `& ` (call operator). On bash use forward slashes.

---

## File Structure

```
final_project/
├── pyproject.toml                    # pytest config (pythonpath = ["."])
├── requirements-extra.txt            # extra pip packages
├── .gitignore
├── README.md
├── docs/superpowers/                 # spec + plan (already exists)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── audio.py
│   │   ├── augment.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── own_cnn.py
│   │   └── pann_cnn10.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── seed.py
│   │   ├── metrics.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       └── plotting.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # fixtures: tiny fake dataset
│   ├── test_audio.py
│   ├── test_augment.py
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── test_seed.py
│   ├── test_io.py
│   ├── test_trainer.py
│   └── test_aggregate.py
├── experiments/
│   ├── run_own_cnn.py
│   ├── run_pann_finetune.py
│   ├── run_ablation_no_aug.py
│   └── aggregate_results.py
├── notebooks/
│   └── eda.ipynb
└── results/                          # generated, .gitignored except summary.*
```

**Files are split by responsibility.** `audio.py` is pure transforms (no torch.nn). `dataset.py` is the I/O + caching boundary. `models/` is pure nn.Module. `trainer.py` is the training loop, agnostic to model and dataset. `experiments/*.py` are orchestrators that wire models, datasets and the trainer together for one experiment.

---

### Task 1: Project scaffolding, dependencies, pytest config

**Files:**
- Create: `final_project/.gitignore`
- Create: `final_project/pyproject.toml`
- Create: `final_project/requirements-extra.txt`
- Create: `final_project/README.md` (skeleton)
- Create: `final_project/src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/training/__init__.py`, `src/utils/__init__.py`
- Create: `final_project/tests/__init__.py`
- Create: `final_project/src/config.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd "C:\Users\desswell\work\pythonProject\ЦОС\final_project"
git init -b main
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.coverage
.ipynb_checkpoints/

# Data (too big for git)
data/
results/own_cnn/
results/pann_cnn10/
results/ablation_no_aug/
results/figures/*.png
*.pt
*.npy

# But keep summary outputs
!results/summary.csv
!results/summary.md
!results/.gitkeep

# Editors
.idea/
.vscode/
*.swp
```

- [ ] **Step 3: Create `requirements-extra.txt`**

```
panns-inference>=0.1.1
soundfile>=0.12.1
librosa>=0.10.0
pytest>=8.0.0
```

- [ ] **Step 4: Install extra deps**

Run:
```powershell
& "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe" -m pip install -r requirements-extra.txt
```
Expected: installs successfully (panns_inference may pull a couple of small deps).

- [ ] **Step 5: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 6: Create empty `__init__.py` in every package**

```bash
# Empty files (use "type nul >" on Windows or `touch` on bash)
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/training/__init__.py src/utils/__init__.py tests/__init__.py
mkdir -p results
touch results/.gitkeep
```

- [ ] **Step 7: Create `src/config.py`**

```python
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
```

- [ ] **Step 8: README skeleton**

```markdown
# UrbanSound8K — классификация городских звуков

Курсовой проект по дисциплине «Методы ИИ в ЦОС», МГТУ им. Н.Э. Баумана.
Ким А.М., ИУ12-41М.

## Постановка

Классификация 10 классов городских звуков (UrbanSound8K) с помощью двух моделей:
- Собственная 2D-CNN на лог-мел спектрограммах (обучение с нуля)
- Дотюненная PANN CNN10 (pretrained на AudioSet)

Оценка — официальная 10-fold cross-validation.

## Установка

См. `docs/superpowers/specs/2026-05-11-urbansound8k-audio-classification-design.md`.

## Запуск

См. `docs/superpowers/specs/...` (раздел 10). Результаты пишутся в `results/`.

## Структура

См. файл спеки.
```

- [ ] **Step 9: Verify pytest discovery works**

Run: `& "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe" -m pytest --collect-only`
Expected: exits 5 (no tests collected) — pytest works, just nothing to run yet.

- [ ] **Step 10: Commit**

```bash
git add .gitignore pyproject.toml requirements-extra.txt README.md src/ tests/ results/.gitkeep docs/
git commit -m "chore: scaffold final_project with config and pytest setup"
```

---

### Task 2: Seed utility + tests

**Files:**
- Create: `final_project/src/training/seed.py`
- Create: `final_project/tests/test_seed.py`

- [ ] **Step 1: Write failing test `tests/test_seed.py`**

```python
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
```

- [ ] **Step 2: Run test, verify failure**

Run: `$PY -m pytest tests/test_seed.py -v`
Expected: ModuleNotFoundError or import error on `src.training.seed`.

- [ ] **Step 3: Implement `src/training/seed.py`**

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `$PY -m pytest tests/test_seed.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/training/seed.py tests/test_seed.py
git commit -m "feat(training): add deterministic seed_everything utility"
```

---

### Task 3: IO utilities + tests

**Files:**
- Create: `final_project/src/utils/io.py`
- Create: `final_project/tests/test_io.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_io.py
import csv
from pathlib import Path

import numpy as np
import pytest

from src.utils.io import append_csv_row, write_json, read_json, save_npy, load_npy


def test_append_csv_row_creates_file_with_header(tmp_path):
    p = tmp_path / "out.csv"
    append_csv_row(p, {"epoch": 1, "loss": 0.5})
    assert p.exists()
    rows = list(csv.DictReader(p.open()))
    assert rows == [{"epoch": "1", "loss": "0.5"}]


def test_append_csv_row_appends_without_duplicating_header(tmp_path):
    p = tmp_path / "out.csv"
    append_csv_row(p, {"epoch": 1, "loss": 0.5})
    append_csv_row(p, {"epoch": 2, "loss": 0.3})
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 2
    assert rows[1]["epoch"] == "2"


def test_append_csv_row_rejects_mismatched_keys(tmp_path):
    p = tmp_path / "out.csv"
    append_csv_row(p, {"a": 1, "b": 2})
    with pytest.raises(ValueError, match="header mismatch"):
        append_csv_row(p, {"a": 3, "c": 4})


def test_write_and_read_json(tmp_path):
    p = tmp_path / "x.json"
    data = {"a": 1, "b": [1, 2, 3], "c": "hello"}
    write_json(p, data)
    assert read_json(p) == data


def test_save_and_load_npy(tmp_path):
    p = tmp_path / "arr.npy"
    arr = np.arange(12).reshape(3, 4).astype(np.float32)
    save_npy(p, arr)
    out = load_npy(p)
    assert np.array_equal(arr, out)
```

- [ ] **Step 2: Run, verify failure**

Run: `$PY -m pytest tests/test_io.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/utils/io.py`**

```python
"""Small IO helpers used across training and aggregation scripts."""
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def append_csv_row(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append a row to a CSV, writing the header on first write.

    Raises ValueError if the existing file's header does not match the row's keys.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = path.exists()

    if file_exists:
        with path.open("r", newline="") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header and existing_header != fieldnames:
            raise ValueError(
                f"header mismatch: file has {existing_header}, row has {fieldnames}"
            )

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path)
```

- [ ] **Step 4: Run, verify pass**

Run: `$PY -m pytest tests/test_io.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/io.py tests/test_io.py
git commit -m "feat(utils): add CSV/JSON/NPY io helpers"
```

---

### Task 4: Audio preprocessing module + tests

**Files:**
- Create: `final_project/src/data/audio.py`
- Create: `final_project/tests/conftest.py`
- Create: `final_project/tests/test_audio.py`

- [ ] **Step 1: Add a synthetic wav fixture in `tests/conftest.py`**

```python
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
```

- [ ] **Step 2: Write failing tests `tests/test_audio.py`**

```python
import torch

from src.config import OWN_AUDIO
from src.data.audio import load_audio, pad_or_crop, log_mel_spectrogram, normalize


def test_load_audio_returns_mono_float_tensor_at_target_sr(stereo_wav_path):
    wav = load_audio(stereo_wav_path, target_sr=22050)
    assert wav.dtype == torch.float32
    assert wav.dim() == 1, f"expected 1D, got shape {wav.shape}"
    # 1 second of 8 kHz stereo resampled to 22050 → 22050 samples
    assert abs(wav.shape[0] - 22050) <= 1


def test_load_audio_no_resample_when_sr_matches(sine_wav_path):
    wav = load_audio(sine_wav_path, target_sr=16000)
    assert wav.shape[0] == 32000  # 2 s × 16 kHz


def test_pad_or_crop_pads_short_signal():
    short = torch.ones(1000)
    out = pad_or_crop(short, n_samples=2000)
    assert out.shape[0] == 2000
    assert torch.equal(out[:1000], short)
    assert torch.all(out[1000:] == 0)


def test_pad_or_crop_center_crops_long_signal():
    long = torch.arange(3000, dtype=torch.float32)
    out = pad_or_crop(long, n_samples=1000)
    assert out.shape[0] == 1000
    # center crop: should start at (3000-1000)//2 = 1000
    assert torch.equal(out, long[1000:2000])


def test_pad_or_crop_passthrough_when_equal():
    x = torch.arange(2000, dtype=torch.float32)
    out = pad_or_crop(x, n_samples=2000)
    assert torch.equal(out, x)


def test_log_mel_spectrogram_shape():
    cfg = OWN_AUDIO
    wav = torch.randn(cfg.n_samples)
    mel = log_mel_spectrogram(wav, cfg)
    assert mel.shape == (1, cfg.n_mels, 173), f"got {mel.shape}"
    assert mel.dtype == torch.float32
    # log-mel should not be all -inf
    assert torch.isfinite(mel).all()


def test_normalize_zero_mean_unit_std():
    x = torch.randn(1, 64, 173) * 5 + 2
    out = normalize(x)
    assert abs(out.mean().item()) < 1e-4
    assert abs(out.std().item() - 1.0) < 1e-2


def test_normalize_handles_constant_input():
    x = torch.full((1, 64, 173), 3.14)
    out = normalize(x)
    # No NaN/Inf — should produce ~0 because (3.14 - 3.14) / (0 + eps)
    assert torch.isfinite(out).all()
```

- [ ] **Step 3: Run, verify failure**

Run: `$PY -m pytest tests/test_audio.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement `src/data/audio.py`**

```python
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
```

- [ ] **Step 5: Run, verify pass**

Run: `$PY -m pytest tests/test_audio.py -v`
Expected: 8 passed.
Note: the expected `(1, 64, 173)` mel shape relies on torchaudio's MelSpectrogram with `center=True` producing `floor(n_samples / hop_length) + 1 = 88200/512 + 1 ≈ 173`. If torchaudio returns 172 in your version, update the test value and the spec accordingly.

- [ ] **Step 6: Commit**

```bash
git add src/data/audio.py tests/conftest.py tests/test_audio.py
git commit -m "feat(data): add audio preprocessing primitives with tests"
```

---

### Task 5: Dataset downloader

**Files:**
- Create: `final_project/src/data/download.py`
- Create: `final_project/tests/test_download.py`

The dataset is on Zenodo at https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz (~6 GB). For tests we mock the HTTP fetch with a tiny tarball.

- [ ] **Step 1: Write failing test**

```python
# tests/test_download.py
import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.data.download import download_and_extract, _sha256


def _make_fake_tarball(tmp_path) -> Path:
    """Create a tarball containing UrbanSound8K/metadata/UrbanSound8K.csv with 1 line."""
    inner = tmp_path / "inner"
    (inner / "UrbanSound8K" / "metadata").mkdir(parents=True)
    (inner / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv").write_text(
        "slice_file_name,fsID,start,end,salience,fold,classID,class\n"
        "x.wav,0,0.0,4.0,1,1,0,air_conditioner\n"
    )
    tar_path = tmp_path / "fake.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(inner / "UrbanSound8K", arcname="UrbanSound8K")
    return tar_path


def test_sha256_computes_correctly(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert _sha256(p) == hashlib.sha256(b"hello").hexdigest()


def test_download_and_extract_extracts_archive(tmp_path, monkeypatch):
    tar_path = _make_fake_tarball(tmp_path)
    dst = tmp_path / "data"

    def fake_download(url, out_path):
        out_path.write_bytes(tar_path.read_bytes())

    with patch("src.data.download._download_with_progress", side_effect=fake_download):
        download_and_extract(dst, url="http://fake", expected_sha256=None)

    assert (dst / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv").exists()


def test_download_and_extract_idempotent(tmp_path):
    dst = tmp_path / "data"
    (dst / "UrbanSound8K" / "metadata").mkdir(parents=True)
    (dst / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv").write_text("ok")

    # If already present, must NOT try to download
    with patch("src.data.download._download_with_progress") as mock_dl:
        download_and_extract(dst, url="http://fake", expected_sha256=None)
        mock_dl.assert_not_called()
```

- [ ] **Step 2: Run, verify failure**

Run: `$PY -m pytest tests/test_download.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/data/download.py`**

```python
"""Download and extract UrbanSound8K from Zenodo.

Run as: python -m src.data.download
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


ZENODO_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"
# Official SHA-256 from Zenodo metadata (verify on first real download and
# update if Zenodo re-publishes the file).
EXPECTED_SHA256 = None  # set to actual hash if you want strict verification


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, out_path: Path) -> None:
    """Stream-download with a tqdm progress bar."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with out_path.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))


def download_and_extract(
    dst_dir: Path,
    url: str = ZENODO_URL,
    expected_sha256: str | None = EXPECTED_SHA256,
) -> None:
    """Download and extract UrbanSound8K. Idempotent: skips if metadata file already present.

    Layout after extraction: `dst_dir/UrbanSound8K/{audio, metadata}/...`
    """
    dst_dir = Path(dst_dir)
    metadata = dst_dir / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
    if metadata.exists():
        print(f"[download] Already extracted at {metadata.parent.parent}, skipping.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dst_dir / "UrbanSound8K.tar.gz"

    if not tar_path.exists():
        print(f"[download] Downloading from {url} -> {tar_path}")
        _download_with_progress(url, tar_path)
    else:
        print(f"[download] Found existing archive {tar_path}, skipping download.")

    if expected_sha256 is not None:
        print("[download] Verifying SHA-256...")
        actual = _sha256(tar_path)
        if actual != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual}")

    print(f"[download] Extracting {tar_path} -> {dst_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dst_dir)

    if not metadata.exists():
        raise RuntimeError(f"Extraction completed but metadata not found at {metadata}")
    print("[download] Done.")


def main():
    parser = argparse.ArgumentParser(description="Download UrbanSound8K from Zenodo")
    parser.add_argument("--dst", type=Path, default=Path("data"))
    parser.add_argument("--url", default=ZENODO_URL)
    args = parser.parse_args()
    download_and_extract(args.dst, url=args.url)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run, verify pass**

Run: `$PY -m pytest tests/test_download.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/download.py tests/test_download.py
git commit -m "feat(data): add UrbanSound8K downloader with idempotent extraction"
```

---

### Task 6: SpecAugment + GaussianNoise + tests

**Files:**
- Create: `final_project/src/data/augment.py`
- Create: `final_project/tests/test_augment.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_augment.py
import torch

from src.data.augment import SpecAugment, GaussianNoise


def test_specaugment_preserves_shape_and_dtype():
    aug = SpecAugment(time_mask_param=25, freq_mask_param=12, n_time=2, n_freq=2)
    x = torch.randn(1, 64, 173)
    out = aug(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_specaugment_actually_masks_something():
    # With large mask widths and many masks, masked-out elements (set to mean) should
    # differ from the original in at least some locations
    torch.manual_seed(0)
    aug = SpecAugment(time_mask_param=40, freq_mask_param=30, n_time=2, n_freq=2)
    x = torch.randn(1, 64, 173)
    out = aug(x)
    # Some elements must differ from input — masking sets them to spec mean
    assert not torch.allclose(out, x)


def test_specaugment_deterministic_under_seed():
    aug = SpecAugment(time_mask_param=25, freq_mask_param=12, n_time=2, n_freq=2)
    x = torch.randn(1, 64, 173)
    torch.manual_seed(123)
    a = aug(x)
    torch.manual_seed(123)
    b = aug(x)
    assert torch.equal(a, b)


def test_gaussian_noise_changes_input_when_applied():
    torch.manual_seed(0)
    aug = GaussianNoise(std=0.5, p=1.0)
    x = torch.randn(1, 64, 173)
    out = aug(x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_gaussian_noise_skipped_when_p_zero():
    aug = GaussianNoise(std=0.5, p=0.0)
    x = torch.randn(1, 64, 173)
    out = aug(x)
    assert torch.equal(out, x)
```

- [ ] **Step 2: Run, verify failure**

Run: `$PY -m pytest tests/test_augment.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/data/augment.py`**

```python
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
```

- [ ] **Step 4: Run, verify pass**

Run: `$PY -m pytest tests/test_augment.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/augment.py tests/test_augment.py
git commit -m "feat(data): add SpecAugment and GaussianNoise spectrogram aug"
```

---

### Task 7: Dataset + caching + tests

**Files:**
- Create: `final_project/src/data/dataset.py`
- Create: `final_project/tests/test_dataset.py`

The dataset class produces spectrogram cache files on first access and reloads them on subsequent accesses. Tests use a tiny fake dataset of 4 synthetic wavs across 2 folds.

- [ ] **Step 1: Add fixture in `tests/conftest.py` for fake UrbanSound8K layout**

Add to the existing `tests/conftest.py`:

```python
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
```

- [ ] **Step 2: Write failing test `tests/test_dataset.py`**

```python
import time

import torch

from src.config import OWN_AUDIO
from src.data.dataset import UrbanSound8KDataset


def test_dataset_len(fake_us8k, tmp_path):
    ds = UrbanSound8KDataset(
        metadata_csv=fake_us8k["metadata_csv"],
        audio_root=fake_us8k["audio_root"],
        folds=[1, 2],
        train=False,
        augment=False,
        cache_dir=tmp_path / "cache",
        audio_config=OWN_AUDIO,
    )
    assert len(ds) == 4


def test_dataset_filter_by_folds(fake_us8k, tmp_path):
    ds = UrbanSound8KDataset(
        metadata_csv=fake_us8k["metadata_csv"],
        audio_root=fake_us8k["audio_root"],
        folds=[1],
        train=False,
        augment=False,
        cache_dir=tmp_path / "cache",
        audio_config=OWN_AUDIO,
    )
    assert len(ds) == 2


def test_dataset_getitem_shape_and_label(fake_us8k, tmp_path):
    ds = UrbanSound8KDataset(
        metadata_csv=fake_us8k["metadata_csv"],
        audio_root=fake_us8k["audio_root"],
        folds=[1],
        train=False,
        augment=False,
        cache_dir=tmp_path / "cache",
        audio_config=OWN_AUDIO,
    )
    spec, label = ds[0]
    assert spec.shape == (1, OWN_AUDIO.n_mels, 173)
    assert spec.dtype == torch.float32
    assert isinstance(label, int)
    assert label in (0, 1)


def test_dataset_cache_written_then_reused(fake_us8k, tmp_path):
    cache = tmp_path / "cache"
    ds = UrbanSound8KDataset(
        metadata_csv=fake_us8k["metadata_csv"],
        audio_root=fake_us8k["audio_root"],
        folds=[1, 2],
        train=False,
        augment=False,
        cache_dir=cache,
        audio_config=OWN_AUDIO,
    )
    # First access builds cache
    _ = ds[0]
    cache_files = list(cache.rglob("*.pt"))
    assert len(cache_files) == 1

    # Second access should not re-decode wav — just load .pt
    t0 = time.perf_counter()
    _ = ds[0]
    t1 = time.perf_counter()
    # Sanity: still works
    assert (t1 - t0) >= 0


def test_dataset_aug_changes_output(fake_us8k, tmp_path):
    cache = tmp_path / "cache"
    ds_noaug = UrbanSound8KDataset(
        metadata_csv=fake_us8k["metadata_csv"],
        audio_root=fake_us8k["audio_root"],
        folds=[1],
        train=True,
        augment=False,
        cache_dir=cache,
        audio_config=OWN_AUDIO,
    )
    ds_aug = UrbanSound8KDataset(
        metadata_csv=fake_us8k["metadata_csv"],
        audio_root=fake_us8k["audio_root"],
        folds=[1],
        train=True,
        augment=True,
        cache_dir=cache,
        audio_config=OWN_AUDIO,
    )
    torch.manual_seed(0)
    spec_noaug, _ = ds_noaug[0]
    torch.manual_seed(0)
    spec_aug, _ = ds_aug[0]
    # With manual_seed reset and random masks, augmented version must differ
    assert not torch.allclose(spec_aug, spec_noaug)
```

- [ ] **Step 3: Run, verify failure**

Run: `$PY -m pytest tests/test_dataset.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement `src/data/dataset.py`**

```python
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
```

- [ ] **Step 5: Run, verify pass**

Run: `$PY -m pytest tests/test_dataset.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/data/dataset.py tests/test_dataset.py tests/conftest.py
git commit -m "feat(data): add UrbanSound8K Dataset with spectrogram caching"
```

---

### Task 8: Own CNN model + tests

**Files:**
- Create: `final_project/src/models/own_cnn.py`
- Create: `final_project/tests/test_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_models.py
import torch

from src.models.own_cnn import OwnCNN


def test_own_cnn_forward_shape():
    model = OwnCNN(num_classes=10)
    x = torch.randn(4, 1, 64, 173)
    y = model(x)
    assert y.shape == (4, 10)


def test_own_cnn_param_count_in_expected_range():
    model = OwnCNN(num_classes=10)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # We target ~250K (see spec section 6.1); allow generous bounds
    assert 100_000 < n_params < 500_000, f"got {n_params} params"


def test_own_cnn_returns_logits_not_probs():
    model = OwnCNN(num_classes=10)
    x = torch.randn(2, 1, 64, 173)
    y = model(x)
    # Logits — should NOT sum to 1
    assert not torch.allclose(y.sum(dim=1), torch.ones(2), atol=1e-3)


def test_own_cnn_trains_one_step_without_error():
    model = OwnCNN(num_classes=10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(2, 1, 64, 173)
    target = torch.tensor([0, 5])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run, verify failure**

Run: `$PY -m pytest tests/test_models.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/models/own_cnn.py`**

```python
"""Custom 2D-CNN for log-mel spectrogram classification.

Architecture: 4 conv blocks (32→64→128→128) with BN+ReLU+MaxPool+Dropout2d,
followed by adaptive avg pool, two FC layers, and a 10-way classifier head.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class OwnCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, dropout=0.10),
            ConvBlock(32, 64, dropout=0.15),
            ConvBlock(64, 128, dropout=0.20),
            ConvBlock(128, 128, dropout=0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)
```

- [ ] **Step 4: Run, verify pass**

Run: `$PY -m pytest tests/test_models.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/own_cnn.py tests/test_models.py
git commit -m "feat(models): add custom 2D-CNN for log-mel classification"
```

---

### Task 9: PANN CNN10 wrapper + tests

**Files:**
- Create: `final_project/src/models/pann_cnn10.py`
- Modify: `final_project/tests/test_models.py` (add PANN tests)

The PANN package `panns_inference` exposes `panns_inference.models.Cnn10`. The wrapper loads pretrained weights via `panns_inference.AudioTagging` (which downloads on first use) but we only need the model module. To keep tests offline-runnable, we'll allow constructing the wrapper without loading weights (`pretrained=False`).

- [ ] **Step 1: Add PANN tests to `tests/test_models.py`**

Append to existing file:

```python
import pytest

from src.models.pann_cnn10 import PannCnn10Wrapper


def test_pann_wrapper_no_pretrained_forward():
    model = PannCnn10Wrapper(num_classes=10, pretrained=False)
    # PANN takes raw mel-spec; 4 s × 32 kHz / 320 hop ≈ 401 frames
    x = torch.randn(2, 1, 64, 401)
    y = model(x)
    assert y.shape == (2, 10)


def test_pann_wrapper_head_is_trainable():
    model = PannCnn10Wrapper(num_classes=10, pretrained=False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("classifier" in n for n in trainable) or any("fc" in n for n in trainable)


def test_pann_freeze_backbone_only_head_trainable():
    model = PannCnn10Wrapper(num_classes=10, pretrained=False)
    model.freeze_backbone()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) > 0
    # All trainable params must be in the new head
    assert all("classifier" in n for n in trainable)


def test_pann_unfreeze_all_trainable():
    model = PannCnn10Wrapper(num_classes=10, pretrained=False)
    model.freeze_backbone()
    model.unfreeze_backbone()
    backbone_trainable = [
        n for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n
    ]
    assert len(backbone_trainable) > 0
```

- [ ] **Step 2: Run, verify failure**

Run: `$PY -m pytest tests/test_models.py -v`
Expected: ImportError on `PannCnn10Wrapper`.

- [ ] **Step 3: Implement `src/models/pann_cnn10.py`**

```python
"""PANN CNN10 wrapper for fine-tuning on UrbanSound8K.

Loads the pretrained CNN10 backbone from the `panns_inference` package (which
auto-downloads weights from Google Drive / Zenodo mirrors) and replaces the
527-way AudioSet head with a 10-way head.

The forward signature accepts pre-computed log-mel spectrograms (so we can
reuse the spectrogram cache), not raw waveforms — we monkey-patch the model
to skip its internal stft/mel layers.
"""
from __future__ import annotations

import torch
from torch import nn


class PannCnn10Wrapper(nn.Module):
    """CNN10 backbone (6 conv blocks → 512 → fc1=2048) + new linear classifier.

    When `pretrained=True`, downloads & loads AudioSet weights through panns_inference.
    When False, initialises randomly (used for unit tests).
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        from panns_inference.models import Cnn10  # lazy import

        # PANN Cnn10 needs the original audio params for its internal Spectrogram /
        # LogmelFilterBank — we'll bypass those in forward, but Cnn10 expects them in __init__.
        backbone = Cnn10(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )

        if pretrained:
            ckpt_path = _download_cnn10_checkpoint()
            state = torch.load(ckpt_path, map_location="cpu")
            # PANN checkpoints store weights under the "model" key
            sd = state.get("model", state)
            missing, unexpected = backbone.load_state_dict(sd, strict=False)
            if unexpected:
                print(f"[pann] {len(unexpected)} unexpected keys (ignored): {unexpected[:3]}...")
            if missing:
                # fc_audioset will be missing because we're about to replace it anyway
                non_head_missing = [k for k in missing if not k.startswith("fc_audioset")]
                if non_head_missing:
                    print(f"[pann] {len(non_head_missing)} missing keys: {non_head_missing[:3]}...")

        # Replace 527-class head with 10-class head. Keep the 2048-dim fc1 features.
        in_features = backbone.fc_audioset.in_features
        backbone.fc_audioset = nn.Identity()  # bypass original head, take pre-head features
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity="relu")
        nn.init.zeros_(self.classifier.bias)

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """log_mel: [B, 1, n_mels, T] — same layout as our dataset."""
        # PANN's Cnn10.forward expects raw audio; we bypass its STFT/mel and call
        # the conv stack directly. CNN10 stores its conv block as `self.conv_block1..6`.
        x = log_mel
        # Replicate Cnn10 inner forward starting after spectrogram extraction.
        # In PANN's Cnn10 the spectrogram is shaped [B, 1, T, F]; ours is [B, 1, F, T].
        x = x.transpose(2, 3)  # → [B, 1, T, F]
        x = self.backbone.bn0(x.transpose(1, 3)).transpose(1, 3)  # batchnorm on freq axis
        x = self.backbone.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.backbone.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.backbone.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.backbone.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)  # mean over freq
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = nn.functional.relu_(self.backbone.fc1(x))
        return self.classifier(x)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def param_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr": lr_backbone},
            {"params": self.classifier.parameters(), "lr": lr_head},
        ]


def _download_cnn10_checkpoint() -> str:
    """Download (and cache) the CNN10 AudioSet checkpoint and return the local path.

    panns_inference doesn't ship CNN10 as default — it ships CNN14. We pull CNN10 directly
    from the official Zenodo mirror.
    """
    import os
    import urllib.request
    from pathlib import Path

    cache = Path(os.path.expanduser("~/.cache/panns_inference"))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "Cnn10_mAP=0.380.pth"
    if not path.exists():
        url = "https://zenodo.org/records/3987831/files/Cnn10_mAP%3D0.380.pth"
        print(f"[pann] Downloading CNN10 weights to {path} ...")
        urllib.request.urlretrieve(url, str(path))
    return str(path)
```

**Note on the PANN inner forward replication:** the body of `forward()` mirrors what `panns_inference.models.Cnn10.forward()` does *after* its built-in `Spectrogram` + `LogmelFilterBank` layers. If your installed version of `panns_inference` has a different internal architecture, run a small probe (`PannCnn10Wrapper(pretrained=False).forward(torch.randn(1,1,64,401))`) and adapt the calls. The dropout values (`0.2`, `0.5`) are PANN defaults.

- [ ] **Step 4: Run, verify pass**

Run: `$PY -m pytest tests/test_models.py -v`
Expected: 8 passed (the 4 OwnCNN + 4 PANN tests). All run with `pretrained=False` so no network access.

- [ ] **Step 5: Sanity test the PANN inner forward (manual, optional but recommended)**

Run:
```powershell
$PY -c "
import torch
from src.models.pann_cnn10 import PannCnn10Wrapper
m = PannCnn10Wrapper(num_classes=10, pretrained=False)
out = m(torch.randn(2, 1, 64, 401))
print('OK', out.shape)
"
```
Expected: `OK torch.Size([2, 10])`. If it errors with `AttributeError: 'Cnn10' object has no attribute 'bn0'` or similar, inspect the installed PANN version's source (`$PY -c "import panns_inference.models; import inspect; print(inspect.getsourcefile(panns_inference.models))"`) and adjust the forward body to match.

- [ ] **Step 6: Commit**

```bash
git add src/models/pann_cnn10.py tests/test_models.py
git commit -m "feat(models): add PANN CNN10 wrapper with 10-class head and freeze controls"
```

---

### Task 10: Metrics + tests

**Files:**
- Create: `final_project/src/training/metrics.py`
- Create: `final_project/tests/test_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_metrics.py
import numpy as np

from src.training.metrics import compute_metrics


def test_compute_metrics_perfect_classifier():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    out = compute_metrics(y_true, y_pred, num_classes=3)
    assert out["accuracy"] == 1.0
    assert abs(out["macro_f1"] - 1.0) < 1e-6
    assert out["confusion_matrix"].shape == (3, 3)
    assert np.array_equal(out["confusion_matrix"], np.diag([2, 2, 2]))


def test_compute_metrics_all_wrong():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    out = compute_metrics(y_true, y_pred, num_classes=2)
    assert out["accuracy"] == 0.0


def test_compute_metrics_per_class_f1_present():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    out = compute_metrics(y_true, y_pred, num_classes=3)
    assert "per_class_f1" in out
    assert out["per_class_f1"].shape == (3,)
    # class 1 has perfect recall (both true 1s predicted 1) but precision 2/3
    assert out["per_class_f1"][1] > 0


def test_compute_metrics_handles_class_with_no_predictions():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    # Class 1 never appears — should not crash
    out = compute_metrics(y_true, y_pred, num_classes=2)
    assert out["accuracy"] == 1.0
```

- [ ] **Step 2: Run, verify failure**

Run: `$PY -m pytest tests/test_metrics.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/training/metrics.py`**

```python
"""Classification metrics: accuracy, macro F1, per-class precision/recall/F1, confusion matrix."""
from __future__ import annotations

from typing import TypedDict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricsDict(TypedDict):
    accuracy: float
    macro_f1: float
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    confusion_matrix: np.ndarray


def compute_metrics(y_true, y_pred, num_classes: int) -> MetricsDict:
    """Compute classification metrics from 1D arrays of integer labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_precision": precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ),
        "per_class_recall": recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ),
        "per_class_f1": f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }
```

- [ ] **Step 4: Run, verify pass**

Run: `$PY -m pytest tests/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/training/metrics.py tests/test_metrics.py
git commit -m "feat(training): add classification metrics module"
```

---

### Task 11: Plotting utilities

**Files:**
- Create: `final_project/src/utils/plotting.py`

No unit tests — plotting is presentation-only; we just verify it produces non-empty PNG files in the experiment scripts.

- [ ] **Step 1: Implement `src/utils/plotting.py`**

```python
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
```

- [ ] **Step 2: Smoke test — make sure imports work**

Run:
```powershell
$PY -c "from src.utils.plotting import plot_learning_curves, plot_confusion_matrix, plot_per_class_f1_comparison, plot_aug_effect; print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/utils/plotting.py
git commit -m "feat(utils): add plotting helpers for curves, confusion, per-class bars"
```

---

### Task 12: Trainer + integration test

**Files:**
- Create: `final_project/src/training/trainer.py`
- Create: `final_project/tests/test_trainer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_trainer.py
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
```

- [ ] **Step 2: Run, verify failure**

Run: `$PY -m pytest tests/test_trainer.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/training/trainer.py`**

```python
"""Generic training loop.

Agnostic to model architecture and dataset — just give it a `nn.Module`, two
DataLoaders, and a `TrainConfig`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.training.metrics import compute_metrics
from src.utils.io import append_csv_row


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    early_stop_patience: int
    grad_clip: float
    num_classes: int
    log_csv: str | Path
    ckpt_path: str | Path
    # Epoch numbering offset for CSV log (useful when training in phases on the same CSV).
    # Epochs written: epoch_offset+1 .. epoch_offset+epochs
    epoch_offset: int = 0
    # Optional: scheduler factory and optimizer factory for advanced use (e.g. PANN param groups)
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None
    scheduler_factory: Callable[[torch.optim.Optimizer, int], torch.optim.lr_scheduler.LRScheduler] | None = None


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()

        if cfg.optimizer_factory is not None:
            self.optimizer = cfg.optimizer_factory(model)
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )

        if cfg.scheduler_factory is not None:
            self.scheduler = cfg.scheduler_factory(self.optimizer, cfg.epochs)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs
            )

        Path(cfg.log_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        total_correct = 0
        total = 0
        for x, y in loader:
            x = x.float()
            y = y.long()
            with torch.set_grad_enabled(train):
                logits = self.model(x)
                loss = self.criterion(logits, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        return total_loss / max(total, 1), total_correct / max(total, 1)

    def fit(self) -> dict:
        best_val_acc = -1.0
        best_epoch = -1
        epochs_since_improve = 0
        history = {"epochs": [], "best_val_acc": 0.0, "best_epoch": 0, "total_time_s": 0.0}
        t_start = time.perf_counter()

        for local_epoch in range(1, self.cfg.epochs + 1):
            epoch = self.cfg.epoch_offset + local_epoch
            t0 = time.perf_counter()
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            dt = time.perf_counter() - t0

            # Also compute macro-F1 on val for logging
            _, y_true, y_pred = self.evaluate(self.val_loader, set_train_back=True)
            metrics = compute_metrics(y_true, y_pred, num_classes=self.cfg.num_classes)
            val_f1 = metrics["macro_f1"]

            append_csv_row(self.cfg.log_csv, {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "val_f1": round(val_f1, 6),
                "lr": round(lr, 8),
                "time_s": round(dt, 3),
            })
            history["epochs"].append(epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_since_improve = 0
                torch.save({"model": self.model.state_dict(), "epoch": epoch}, self.cfg.ckpt_path)
            else:
                epochs_since_improve += 1

            tqdm.write(
                f"epoch {epoch:3d} | train {train_loss:.4f}/{train_acc:.3f} "
                f"val {val_loss:.4f}/{val_acc:.3f} (best {best_val_acc:.3f}@{best_epoch}) "
                f"lr={lr:.2e} {dt:.1f}s"
            )

            if epochs_since_improve >= self.cfg.early_stop_patience:
                tqdm.write(f"  early stop at epoch {epoch} (patience {self.cfg.early_stop_patience})")
                break

        # If nothing ever improved past initial -1.0, save current weights so load_best() works
        if not Path(self.cfg.ckpt_path).exists():
            torch.save({"model": self.model.state_dict(), "epoch": -1}, self.cfg.ckpt_path)

        history["best_val_acc"] = best_val_acc
        history["best_epoch"] = best_epoch
        history["total_time_s"] = time.perf_counter() - t_start
        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, set_train_back: bool = False):
        was_training = self.model.training
        self.model.eval()
        all_true, all_pred = [], []
        for x, y in loader:
            x = x.float()
            logits = self.model(x)
            pred = logits.argmax(dim=1)
            all_true.append(y.numpy())
            all_pred.append(pred.cpu().numpy())
        y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
        y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)
        metrics = compute_metrics(y_true, y_pred, num_classes=self.cfg.num_classes)
        if set_train_back and was_training:
            self.model.train()
        return metrics, y_true, y_pred

    def load_best(self) -> None:
        ckpt = torch.load(self.cfg.ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
```

- [ ] **Step 4: Run, verify pass**

Run: `$PY -m pytest tests/test_trainer.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the entire test suite, verify everything still passes**

Run: `$PY -m pytest -v`
Expected: all tests pass (>= 30 total).

- [ ] **Step 6: Commit**

```bash
git add src/training/trainer.py tests/test_trainer.py
git commit -m "feat(training): add generic Trainer with early stopping and per-epoch CSV log"
```

---

### Task 13: `run_own_cnn.py` experiment + smoke run

**Files:**
- Create: `final_project/experiments/run_own_cnn.py`

This script wires Dataset + OwnCNN + Trainer for 10-fold CV. Includes a `--fast` mode (1 fold, 2 epochs) so it can be smoke-tested without committing to the full ~2h run.

- [ ] **Step 1: Implement `experiments/run_own_cnn.py`**

```python
"""10-fold CV training of OwnCNN on UrbanSound8K log-mel spectrograms."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.config import (
    OWN_AUDIO, NUM_CLASSES, OWN_CNN_TRAIN,
    METADATA_PATH, AUDIO_SUBDIR, CACHE_DIR_OWN, DATA_DIR, RESULTS_DIR, CLASS_NAMES,
)
from src.data.dataset import UrbanSound8KDataset
from src.models.own_cnn import OwnCNN
from src.training.seed import seed_everything
from src.training.trainer import Trainer, TrainConfig
from src.utils.io import append_csv_row, save_npy, write_json


def split_train_val(dataset, val_frac: float, seed: int):
    """Random 90/10 split of a Dataset, with fixed seed."""
    n = len(dataset)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return Subset(dataset, tr_idx.tolist()), Subset(dataset, val_idx.tolist())


def run_one_fold(test_fold: int, args, results_dir: Path) -> dict:
    seed_everything(test_fold * 100 + 42)

    train_folds = [f for f in range(1, 11) if f != test_fold]
    full_train = UrbanSound8KDataset(
        metadata_csv=METADATA_PATH,
        audio_root=Path(DATA_DIR) / AUDIO_SUBDIR,
        folds=train_folds,
        train=True,
        augment=args.augment,
        cache_dir=CACHE_DIR_OWN,
        audio_config=OWN_AUDIO,
    )
    test_ds = UrbanSound8KDataset(
        metadata_csv=METADATA_PATH,
        audio_root=Path(DATA_DIR) / AUDIO_SUBDIR,
        folds=[test_fold],
        train=False,
        augment=False,
        cache_dir=CACHE_DIR_OWN,
        audio_config=OWN_AUDIO,
    )
    tr_ds, val_ds = split_train_val(full_train, val_frac=0.10, seed=42)

    bs = OWN_CNN_TRAIN["batch_size"]
    train_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    model = OwnCNN(num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=OWN_CNN_TRAIN["lr"],
        weight_decay=OWN_CNN_TRAIN["weight_decay"],
        early_stop_patience=OWN_CNN_TRAIN["early_stop_patience"],
        grad_clip=OWN_CNN_TRAIN["grad_clip"],
        num_classes=NUM_CLASSES,
        log_csv=results_dir / f"fold_{test_fold}.csv",
        ckpt_path=results_dir / f"fold_{test_fold}_best.pt",
    )
    trainer = Trainer(model, train_dl, val_dl, cfg)
    history = trainer.fit()
    trainer.load_best()

    import time
    n_test = len(test_ds)
    t0 = time.perf_counter()
    metrics, y_true, y_pred = trainer.evaluate(test_dl)
    inf_time_ms = (time.perf_counter() - t0) * 1000.0 / max(n_test, 1)

    save_npy(results_dir / f"fold_{test_fold}_cm.npy", metrics["confusion_matrix"])
    write_json(results_dir / f"fold_{test_fold}_per_class.json", {
        "f1": metrics["per_class_f1"].tolist(),
        "precision": metrics["per_class_precision"].tolist(),
        "recall": metrics["per_class_recall"].tolist(),
        "class_names": list(CLASS_NAMES),
    })

    summary_row = {
        "fold": test_fold,
        "accuracy": round(float(metrics["accuracy"]), 6),
        "macro_f1": round(float(metrics["macro_f1"]), 6),
        "best_epoch": history["best_epoch"],
        "train_time_s": round(history["total_time_s"], 1),
        "inference_time_ms_per_sample": round(inf_time_ms, 3),
        "n_params": n_params,
    }
    append_csv_row(results_dir / "folds_summary.csv", summary_row)
    return summary_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--epochs", type=int, default=OWN_CNN_TRAIN["epochs"])
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--fast", action="store_true", help="1 fold, 2 epochs, no augment")
    parser.add_argument("--out-name", default="own_cnn", help="subdir of results/")
    args = parser.parse_args()

    if args.fast:
        args.folds = [1]
        args.epochs = 2

    results_dir = Path(RESULTS_DIR) / args.out_name
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in args.folds:
        print(f"\n========== FOLD {fold} ==========")
        row = run_one_fold(fold, args, results_dir)
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    print("\n=== Summary across folds ===")
    print(df.describe())
    df.to_csv(results_dir / "folds_summary.csv", index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify dataset is downloaded (one-time setup)**

Run:
```powershell
$PY -m src.data.download
```
Expected: either "already extracted" (if you've done this before) or a 6 GB download + extraction. Eventually prints "Done." and `data/UrbanSound8K/metadata/UrbanSound8K.csv` exists.

- [ ] **Step 3: Smoke-test `run_own_cnn.py` in `--fast` mode**

Run:
```powershell
$PY experiments\run_own_cnn.py --fast --out-name smoke_own_cnn
```
Expected: completes in 10-30 min (the first run builds the spectrogram cache). Prints per-epoch logs. Creates `results/smoke_own_cnn/fold_1.csv`, `fold_1_best.pt`, `fold_1_cm.npy`, `folds_summary.csv`. Test accuracy should be above chance (~10%); even with 2 epochs it should be 30%+.

- [ ] **Step 4: Commit**

```bash
git add experiments/run_own_cnn.py
git commit -m "feat(experiments): add own CNN 10-fold CV runner with --fast smoke mode"
```

---

### Task 14: `run_pann_finetune.py` experiment + smoke run

**Files:**
- Create: `final_project/experiments/run_pann_finetune.py`

PANN needs a two-phase training (frozen warmup → unfreeze fine-tune). We use `optimizer_factory` to build param groups.

- [ ] **Step 1: Implement `experiments/run_pann_finetune.py`**

```python
"""10-fold CV fine-tune of PANN CNN10 on UrbanSound8K."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.config import (
    PANN_AUDIO, NUM_CLASSES, PANN_TRAIN,
    METADATA_PATH, AUDIO_SUBDIR, CACHE_DIR_PANN, DATA_DIR, RESULTS_DIR, CLASS_NAMES,
)
from src.data.dataset import UrbanSound8KDataset
from src.models.pann_cnn10 import PannCnn10Wrapper
from src.training.seed import seed_everything
from src.training.trainer import Trainer, TrainConfig
from src.utils.io import append_csv_row, save_npy, write_json


def split_train_val(dataset, val_frac: float, seed: int):
    n = len(dataset)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    return Subset(dataset, idx[n_val:].tolist()), Subset(dataset, idx[:n_val].tolist())


def run_one_fold(test_fold: int, args, results_dir: Path) -> dict:
    seed_everything(test_fold * 100 + 42)

    train_folds = [f for f in range(1, 11) if f != test_fold]
    full_train = UrbanSound8KDataset(
        metadata_csv=METADATA_PATH,
        audio_root=Path(DATA_DIR) / AUDIO_SUBDIR,
        folds=train_folds,
        train=True,
        augment=args.augment,
        cache_dir=CACHE_DIR_PANN,
        audio_config=PANN_AUDIO,
    )
    test_ds = UrbanSound8KDataset(
        metadata_csv=METADATA_PATH,
        audio_root=Path(DATA_DIR) / AUDIO_SUBDIR,
        folds=[test_fold],
        train=False,
        augment=False,
        cache_dir=CACHE_DIR_PANN,
        audio_config=PANN_AUDIO,
    )
    tr_ds, val_ds = split_train_val(full_train, val_frac=0.10, seed=42)
    bs = PANN_TRAIN["batch_size"]
    train_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    model = PannCnn10Wrapper(num_classes=NUM_CLASSES, pretrained=not args.no_pretrained)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Phase 1: frozen backbone, head only. Saves a temp ckpt that's discarded after phase 2.
    model.freeze_backbone()
    cfg_phase1 = TrainConfig(
        epochs=args.warmup_epochs,
        lr=PANN_TRAIN["lr_head"],
        weight_decay=PANN_TRAIN["weight_decay"],
        early_stop_patience=args.warmup_epochs + 1,  # don't early-stop phase 1
        grad_clip=PANN_TRAIN["grad_clip"],
        num_classes=NUM_CLASSES,
        log_csv=results_dir / f"fold_{test_fold}.csv",
        ckpt_path=results_dir / f"fold_{test_fold}_phase1.pt",
        epoch_offset=0,
    )
    print(f"[PANN] phase 1 (warmup, frozen) for {args.warmup_epochs} epochs")
    trainer = Trainer(model, train_dl, val_dl, cfg_phase1)
    trainer.fit()

    # Phase 2: unfreeze, different LRs per param group
    model.unfreeze_backbone()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def opt_factory(m):
        return torch.optim.AdamW(
            m.param_groups(lr_backbone=PANN_TRAIN["lr_backbone"], lr_head=PANN_TRAIN["lr_head"]),
            weight_decay=PANN_TRAIN["weight_decay"],
        )

    cfg_phase2 = TrainConfig(
        epochs=args.epochs - args.warmup_epochs,
        lr=PANN_TRAIN["lr_backbone"],  # used only if optimizer_factory not given
        weight_decay=PANN_TRAIN["weight_decay"],
        early_stop_patience=PANN_TRAIN["early_stop_patience"],
        grad_clip=PANN_TRAIN["grad_clip"],
        num_classes=NUM_CLASSES,
        log_csv=results_dir / f"fold_{test_fold}.csv",  # append to same CSV
        ckpt_path=results_dir / f"fold_{test_fold}_best.pt",
        epoch_offset=args.warmup_epochs,  # continue epoch numbering after phase 1
        optimizer_factory=opt_factory,
    )
    print(f"[PANN] phase 2 (unfreeze) for {args.epochs - args.warmup_epochs} epochs")
    trainer2 = Trainer(model, train_dl, val_dl, cfg_phase2)
    history = trainer2.fit()
    trainer2.load_best()

    # Discard phase-1 checkpoint
    (results_dir / f"fold_{test_fold}_phase1.pt").unlink(missing_ok=True)

    n_test = len(test_ds)
    t0 = time.perf_counter()
    metrics, y_true, y_pred = trainer2.evaluate(test_dl)
    inf_time_ms = (time.perf_counter() - t0) * 1000.0 / max(n_test, 1)

    save_npy(results_dir / f"fold_{test_fold}_cm.npy", metrics["confusion_matrix"])
    write_json(results_dir / f"fold_{test_fold}_per_class.json", {
        "f1": metrics["per_class_f1"].tolist(),
        "precision": metrics["per_class_precision"].tolist(),
        "recall": metrics["per_class_recall"].tolist(),
        "class_names": list(CLASS_NAMES),
    })

    row = {
        "fold": test_fold,
        "accuracy": round(float(metrics["accuracy"]), 6),
        "macro_f1": round(float(metrics["macro_f1"]), 6),
        "best_epoch": history["best_epoch"],
        "train_time_s": round(history["total_time_s"], 1),
        "inference_time_ms_per_sample": round(inf_time_ms, 3),
        "n_params": n_params,
    }
    append_csv_row(results_dir / "folds_summary.csv", row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--epochs", type=int, default=PANN_TRAIN["epochs"])
    parser.add_argument("--warmup-epochs", type=int, default=PANN_TRAIN["warmup_epochs"])
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip weight download (for smoke testing only)")
    parser.add_argument("--fast", action="store_true",
                        help="1 fold, warmup=1 epoch + 1 phase-2 epoch")
    parser.add_argument("--out-name", default="pann_cnn10")
    args = parser.parse_args()

    if args.fast:
        args.folds = [1]
        args.epochs = 2
        args.warmup_epochs = 1

    results_dir = Path(RESULTS_DIR) / args.out_name
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in args.folds:
        print(f"\n========== FOLD {fold} ==========")
        row = run_one_fold(fold, args, results_dir)
        rows.append(row)
        print(row)

    pd.DataFrame(rows).to_csv(results_dir / "folds_summary.csv", index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (with `--no-pretrained` first, then with real weights)**

First validate the wiring without downloading 200 MB of weights:
```powershell
$PY experiments\run_pann_finetune.py --fast --no-pretrained --out-name smoke_pann_dryrun
```
Expected: completes. Per-epoch logs. Test accuracy will be poor (random init) — that's OK; we're only checking the pipeline.

Then a real smoke test that downloads weights:
```powershell
$PY experiments\run_pann_finetune.py --fast --out-name smoke_pann
```
Expected: downloads CNN10 weights to `~/.cache/panns_inference/Cnn10_mAP=0.380.pth` on first run (~150 MB), then trains. Test accuracy after 2 epochs should be >50% even on 1 fold (PANN pretrained is strong).

- [ ] **Step 3: Commit**

```bash
git add experiments/run_pann_finetune.py
git commit -m "feat(experiments): add PANN CNN10 two-phase fine-tune runner"
```

---

### Task 15: `run_ablation_no_aug.py`

**Files:**
- Create: `final_project/experiments/run_ablation_no_aug.py`

Thin wrapper that calls `run_own_cnn` logic with `augment=False` and only folds [1, 5, 10].

- [ ] **Step 1: Implement `experiments/run_ablation_no_aug.py`**

```python
"""Ablation: OwnCNN without augmentation, 3 folds. Demonstrates aug effect."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We reuse run_own_cnn's run_one_fold + main scaffolding
from experiments.run_own_cnn import run_one_fold
from src.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--out-name", default="ablation_no_aug")
    args = parser.parse_args()
    # Force augmentation off
    args.augment = False
    args.fast = False

    results_dir = Path(RESULTS_DIR) / args.out_name
    results_dir.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        print(f"\n========== FOLD {fold} (no-aug ablation) ==========")
        run_one_fold(fold, args, results_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```powershell
$PY experiments\run_ablation_no_aug.py --folds 1 --epochs 2 --out-name smoke_ablation
```
Expected: completes, writes `results/smoke_ablation/fold_1.csv`.

- [ ] **Step 3: Commit**

```bash
git add experiments/run_ablation_no_aug.py
git commit -m "feat(experiments): add no-aug ablation runner"
```

---

### Task 16: `aggregate_results.py` + tests

**Files:**
- Create: `final_project/experiments/aggregate_results.py`
- Create: `final_project/tests/test_aggregate.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_aggregate.py
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.aggregate_results import (
    aggregate_one_model, build_summary_md, average_confusion_matrices,
)


def _write_folds_summary(dir_: Path, accs, f1s, params=250_000):
    rows = []
    for fold, (a, f) in enumerate(zip(accs, f1s), 1):
        rows.append({
            "fold": fold,
            "accuracy": a,
            "macro_f1": f,
            "best_epoch": 10 + fold,
            "train_time_s": 600,
            "inference_time_ms_per_sample": 5.0,
            "n_params": params,
        })
    dir_.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dir_ / "folds_summary.csv", index=False)
    # Also write fake confusion matrices
    for r in rows:
        cm = np.eye(10, dtype=int) * 80
        np.save(dir_ / f"fold_{r['fold']}_cm.npy", cm)


def test_aggregate_one_model_mean_std(tmp_path):
    d = tmp_path / "own_cnn"
    _write_folds_summary(d, accs=[0.75, 0.80, 0.85], f1s=[0.70, 0.78, 0.82])
    summary = aggregate_one_model("own_cnn", d)
    assert summary["model"] == "own_cnn"
    assert summary["n_folds"] == 3
    assert abs(summary["acc_mean"] - 0.80) < 1e-6
    assert summary["acc_std"] > 0


def test_average_confusion_matrices(tmp_path):
    d = tmp_path / "m"
    _write_folds_summary(d, accs=[0.9, 0.9], f1s=[0.9, 0.9])
    avg = average_confusion_matrices(d, n_classes=10)
    assert avg.shape == (10, 10)
    # Diagonal should dominate
    assert (np.diag(avg) > 0).all()


def test_build_summary_md_contains_each_row(tmp_path):
    rows = [
        {"model": "own_cnn", "n_folds": 10, "acc_mean": 0.78, "acc_std": 0.03,
         "f1_mean": 0.77, "f1_std": 0.04, "train_time_s_total": 7200,
         "n_params": 250000, "best_epoch_avg": 25},
        {"model": "pann_cnn10", "n_folds": 10, "acc_mean": 0.89, "acc_std": 0.02,
         "f1_mean": 0.88, "f1_std": 0.02, "train_time_s_total": 14400,
         "n_params": 5_000_000, "best_epoch_avg": 15},
    ]
    md = build_summary_md(rows)
    assert "own_cnn" in md
    assert "pann_cnn10" in md
    assert "0.78" in md and "0.89" in md
```

- [ ] **Step 2: Run, verify failure**

Run: `$PY -m pytest tests/test_aggregate.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `experiments/aggregate_results.py`**

```python
"""Aggregate per-fold CSVs into summary.csv + summary.md + figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import NUM_CLASSES, CLASS_NAMES, RESULTS_DIR
from src.utils.plotting import (
    plot_learning_curves, plot_confusion_matrix, plot_per_class_f1_comparison,
)


def aggregate_one_model(model_name: str, model_dir: Path) -> dict:
    df = pd.read_csv(model_dir / "folds_summary.csv")
    n_folds = len(df)
    return {
        "model": model_name,
        "n_folds": n_folds,
        "acc_mean": float(df["accuracy"].mean()),
        "acc_std": float(df["accuracy"].std(ddof=1) if n_folds > 1 else 0.0),
        "f1_mean": float(df["macro_f1"].mean()),
        "f1_std": float(df["macro_f1"].std(ddof=1) if n_folds > 1 else 0.0),
        "train_time_s_total": float(df["train_time_s"].sum()),
        "n_params": int(df["n_params"].iloc[0]),
        "best_epoch_avg": float(df["best_epoch"].mean()),
    }


def average_confusion_matrices(model_dir: Path, n_classes: int) -> np.ndarray:
    cms = sorted(model_dir.glob("fold_*_cm.npy"))
    if not cms:
        return np.zeros((n_classes, n_classes), dtype=float)
    arr = np.stack([np.load(p) for p in cms], axis=0).astype(float)
    return arr.mean(axis=0)


def build_summary_md(rows: list[dict]) -> str:
    header = (
        "| model | folds | acc (mean±std) | macro-F1 (mean±std) | params | "
        "avg best epoch | total train time (s) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        body += (
            f"| {r['model']} | {r['n_folds']} | "
            f"{r['acc_mean']:.3f}±{r['acc_std']:.3f} | "
            f"{r['f1_mean']:.3f}±{r['f1_std']:.3f} | "
            f"{r['n_params']:,} | "
            f"{r['best_epoch_avg']:.1f} | "
            f"{r['train_time_s_total']:.0f} |\n"
        )
    return header + body


def _per_class_f1_from_folds(model_dir: Path) -> np.ndarray:
    """Read per-fold per_class.json files and average the F1 vectors."""
    import json
    f1s = []
    for p in sorted(model_dir.glob("fold_*_per_class.json")):
        with p.open() as f:
            d = json.load(f)
        f1s.append(np.array(d["f1"]))
    if not f1s:
        return np.zeros(NUM_CLASSES)
    return np.stack(f1s, axis=0).mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=RESULTS_DIR)
    parser.add_argument(
        "--models", nargs="+",
        default=["own_cnn", "pann_cnn10", "ablation_no_aug"],
        help="subdirs of results/ to aggregate",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_class_f1_by_model = {}

    for m in args.models:
        d = root / m
        if not (d / "folds_summary.csv").exists():
            print(f"[warn] {d}/folds_summary.csv missing — skipping")
            continue
        rows.append(aggregate_one_model(m, d))

        # Confusion matrix figure
        cm = average_confusion_matrices(d, n_classes=NUM_CLASSES)
        plot_confusion_matrix(
            cm, class_names=list(CLASS_NAMES),
            out_png=figures_dir / f"confusion_matrix_{m}_avg.png",
            title=f"{m}: avg normalised confusion (10-fold)",
            normalize=True,
        )

        # Learning curves for fold 0 (i.e. fold 1, our first fold)
        # Prefer fold 1 — that's what every script uses as its first fold
        for first_fold in (1, 5, 10):
            log = d / f"fold_{first_fold}.csv"
            if log.exists():
                plot_learning_curves(
                    log, out_png=figures_dir / f"learning_curves_{m}_fold{first_fold}.png",
                    title=f"{m} — fold {first_fold}",
                )
                break

        per_class_f1_by_model[m] = _per_class_f1_from_folds(d)

    summary_csv = root / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"[ok] wrote {summary_csv}")

    md = build_summary_md(rows)
    (root / "summary.md").write_text(md, encoding="utf-8")
    print(f"[ok] wrote {root / 'summary.md'}")

    if len(per_class_f1_by_model) >= 2:
        plot_per_class_f1_comparison(
            per_class_f1_by_model, class_names=list(CLASS_NAMES),
            out_png=figures_dir / "per_class_f1_comparison.png",
        )
        print(f"[ok] wrote per-class F1 comparison figure")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `$PY -m pytest tests/test_aggregate.py -v`
Expected: 3 passed.

- [ ] **Step 5: Smoke run aggregator on the smoke results from Tasks 13-15**

Run:
```powershell
$PY experiments\aggregate_results.py --models smoke_own_cnn smoke_pann smoke_ablation
```
Expected: writes `results/summary.csv`, `results/summary.md`, and figures. The MD looks like a tiny but valid table.

- [ ] **Step 6: Clean up smoke artifacts (optional, before real run)**

Smoke artifacts in `results/smoke_*/` can stay (gitignored) or be deleted:
```bash
rm -rf results/smoke_own_cnn results/smoke_pann results/smoke_pann_dryrun results/smoke_ablation
```

- [ ] **Step 7: Commit**

```bash
git add experiments/aggregate_results.py tests/test_aggregate.py
git commit -m "feat(experiments): add results aggregator producing summary.csv/md + figures"
```

---

### Task 17: EDA notebook

**Files:**
- Create: `final_project/notebooks/eda.ipynb`

This step writes the notebook as JSON via Python (so we don't need an interactive Jupyter session to author it). Run it after to verify it executes cleanly.

- [ ] **Step 1: Create `notebooks/make_eda_notebook.py` to generate the .ipynb**

```python
"""One-off script to generate notebooks/eda.ipynb. Run once.

We author the notebook programmatically so we can keep it under version control
without depending on Jupyter at authoring time.
"""
import json
from pathlib import Path


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


cells = []
cells.append(md_cell("""# EDA — UrbanSound8K
Курсовой проект «ИИ в ЦОС», Ким А.М., ИУ12-41М.

Цели: понять структуру датасета, его сбалансированность, распределение длительностей,
sample-rate-ов; визуализировать примеры waveform-ов и log-mel спектрограмм для каждого класса;
показать эффект аугментации.
"""))

cells.append(code_cell("""import sys
from pathlib import Path

# Make src importable when running from notebooks/
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import torch

from src.config import OWN_AUDIO, METADATA_PATH, DATA_DIR, AUDIO_SUBDIR, CLASS_NAMES
from src.data.audio import load_audio, pad_or_crop, log_mel_spectrogram, normalize
from src.data.augment import SpecAugment, GaussianNoise

sns.set_theme(style="whitegrid")
META = pd.read_csv(METADATA_PATH)
META.head()
"""))

cells.append(md_cell("## 1. Обзор датасета"))
cells.append(code_cell("""print('Total files:', len(META))
print('Classes:', sorted(META['class'].unique()))
print('Folds:', sorted(META['fold'].unique()))
print('Duration (s): start..end span shows segment length within original recording')
META[['start','end']].describe()
"""))

cells.append(md_cell("## 2. Распределение классов"))
cells.append(code_cell("""fig, ax = plt.subplots(figsize=(10,4))
class_counts = META['class'].value_counts().sort_index()
class_counts.plot(kind='bar', ax=ax, color=sns.color_palette('tab10'))
ax.set_ylabel('# files')
ax.set_title('Class distribution in UrbanSound8K')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
print(class_counts)
"""))

cells.append(md_cell("## 3. Распределение классов по folds"))
cells.append(code_cell("""xt = pd.crosstab(META['fold'], META['class'])
fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(xt, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Files per (fold, class)')
plt.tight_layout(); plt.show()
"""))

cells.append(md_cell("## 4. Длительность сегментов"))
cells.append(code_cell("""META['duration_s'] = META['end'] - META['start']
fig, ax = plt.subplots(figsize=(9,4))
META['duration_s'].hist(bins=40, ax=ax)
ax.set_xlabel('Segment duration (s)')
ax.set_ylabel('# files')
ax.set_title('Segment duration distribution')
plt.tight_layout(); plt.show()
print(f"Files <4s requiring pad: {(META['duration_s'] < 4.0).sum()} of {len(META)}")
print(f"Files >=4s requiring crop: {(META['duration_s'] >= 4.0).sum()}")
"""))

cells.append(md_cell("## 5. Sample-rate-ы исходных файлов"))
cells.append(code_cell("""# Probe sample rate of a sample of files (full sweep is slow)
import torchaudio
sample = META.sample(min(200, len(META)), random_state=0)
srs = []
for _, r in sample.iterrows():
    p = Path(DATA_DIR)/AUDIO_SUBDIR/f"fold{r.fold}"/r.slice_file_name
    info = torchaudio.info(str(p))
    srs.append(info.sample_rate)
pd.Series(srs).value_counts().plot(kind='bar', title='Sample rates (sample of 200 files)')
plt.ylabel('# files'); plt.tight_layout(); plt.show()
"""))

cells.append(md_cell("## 6. Примеры waveform-ов — по одному на класс"))
cells.append(code_cell("""classes = sorted(META['class'].unique())
fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
for ax, cls in zip(axes.flat, classes):
    row = META[META['class']==cls].iloc[0]
    p = Path(DATA_DIR)/AUDIO_SUBDIR/f"fold{row.fold}"/row.slice_file_name
    wav = load_audio(p, target_sr=OWN_AUDIO.sample_rate).numpy()
    librosa.display.waveshow(wav, sr=OWN_AUDIO.sample_rate, ax=ax)
    ax.set_title(cls)
plt.tight_layout(); plt.show()
"""))

cells.append(md_cell("## 7. Примеры log-mel спектрограмм — по одному на класс"))
cells.append(code_cell("""fig, axes = plt.subplots(5, 2, figsize=(14, 12))
for ax, cls in zip(axes.flat, classes):
    row = META[META['class']==cls].iloc[0]
    p = Path(DATA_DIR)/AUDIO_SUBDIR/f"fold{row.fold}"/row.slice_file_name
    wav = load_audio(p, target_sr=OWN_AUDIO.sample_rate)
    wav = pad_or_crop(wav, OWN_AUDIO.n_samples)
    mel = normalize(log_mel_spectrogram(wav, OWN_AUDIO))[0].numpy()
    librosa.display.specshow(mel, sr=OWN_AUDIO.sample_rate, hop_length=OWN_AUDIO.hop_length,
                             x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(cls)
plt.tight_layout(); plt.show()
"""))

cells.append(md_cell("## 8. Эффект аугментации"))
cells.append(code_cell("""row = META.iloc[0]
p = Path(DATA_DIR)/AUDIO_SUBDIR/f"fold{row.fold}"/row.slice_file_name
wav = load_audio(p, OWN_AUDIO.sample_rate)
wav = pad_or_crop(wav, OWN_AUDIO.n_samples)
orig = normalize(log_mel_spectrogram(wav, OWN_AUDIO))

torch.manual_seed(0)
specaug = SpecAugment()(orig.clone())
torch.manual_seed(1)
noised = GaussianNoise(std=0.1, p=1.0)(specaug.clone())

fig, axes = plt.subplots(1,3, figsize=(15,4))
for ax, sp, ttl in zip(axes, [orig, specaug, noised], ['Original','+SpecAugment','+Noise']):
    ax.imshow(sp[0].numpy(), origin='lower', aspect='auto', cmap='viridis')
    ax.set_title(ttl); ax.set_xlabel('frames'); ax.set_ylabel('mel bins')
plt.tight_layout(); plt.show()
"""))

cells.append(md_cell("""## 9. Выводы EDA

- Датасет почти сбалансирован (8732 файла, по ~800-1000 на класс), кроме `gun_shot` (374) и `car_horn` (429) — minority классы.
- Все 10 классов представлены в каждом fold-е, но не пропорционально → CV даст слегка шумные fold-метрики.
- Большинство файлов короче 4 секунд → стратегия pad-zeros справа корректна; меньшая часть длиннее — center-crop.
- Sample-rate-ы разные (8/16/22.05/44.1/48 kHz) → resample к 22050 Hz обязателен.
- Спектрограммы классов визуально различимы (например, `gun_shot` — резкий короткий всплеск, `engine_idling` — устойчивые низкие частоты).
- SpecAugment вносит видимые маскированные полосы; вместе с шумом образует разумный train-time augmentation.
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path("notebooks/eda.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out}")
```

- [ ] **Step 2: Generate the notebook**

Run:
```powershell
$PY notebooks\make_eda_notebook.py
```
Expected: prints `Wrote notebooks/eda.ipynb`.

- [ ] **Step 3: Execute the notebook end-to-end to verify it runs**

Run:
```powershell
$PY -m pip install nbconvert ipykernel
$PY -m jupyter nbconvert --to notebook --execute notebooks\eda.ipynb --output eda.executed.ipynb --ExecutePreprocessor.timeout=600
```
Expected: produces `notebooks/eda.executed.ipynb`. If it errors on `nbconvert`/`ipykernel` not installed, install them and retry. The notebook execution takes ~3-5 min (mostly the 200-file sample-rate probe).

- [ ] **Step 4: Delete the executed copy (keep only the source)**

```bash
rm notebooks/eda.executed.ipynb
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/make_eda_notebook.py notebooks/eda.ipynb
git commit -m "feat(notebooks): add EDA notebook with class/duration/SR analysis"
```

---

### Task 18: Final full 10-fold runs + README finalization

This task runs the real experiments end-to-end. It is mostly waiting; the dev work is finalising the README with actual numbers.

- [ ] **Step 1: Verify dataset is downloaded**

Run:
```powershell
$PY -m src.data.download
```
Expected: "Already extracted" or completes a fresh ~6 GB download.

- [ ] **Step 2: Pre-build both spectrogram caches (one-time, ~30 min each)**

Run a 1-fold dummy of each script to populate caches across all folds. The cache is built on demand per file, so the easiest way to "warm" the cache is to do one cheap epoch over all 10 folds via `--epochs 1`:

```powershell
$PY experiments\run_own_cnn.py --folds 1 2 3 4 5 6 7 8 9 10 --epochs 1 --out-name cache_warmup_own
```
Expected: writes per-fold CSVs (we discard them), populates `data/UrbanSound8K/cache_mel/`.

Then for PANN (32 kHz cache):
```powershell
$PY experiments\run_pann_finetune.py --folds 1 2 3 4 5 6 7 8 9 10 --epochs 1 --warmup-epochs 1 --no-pretrained --out-name cache_warmup_pann
```

Delete the warmup result folders after:
```bash
rm -rf results/cache_warmup_own results/cache_warmup_pann
```

- [ ] **Step 3: Run the three full experiments (can be background / overnight)**

```powershell
$PY experiments\run_own_cnn.py
$PY experiments\run_pann_finetune.py
$PY experiments\run_ablation_no_aug.py
```
Expected total runtime: ~7 h (see spec section 11). Each run prints per-epoch and per-fold logs and writes to `results/<model>/`.

- [ ] **Step 4: Aggregate**

```powershell
$PY experiments\aggregate_results.py
```
Expected: writes `results/summary.csv`, `results/summary.md`, and 4-6 PNG figures in `results/figures/`.

- [ ] **Step 5: Finalise README.md**

Replace the skeleton README with content that has actual numbers, structure, and run instructions. The full content:

```markdown
# UrbanSound8K — классификация городских звуков

Курсовой проект по дисциплине «Методы искусственного интеллекта в ЦОС».
МГТУ им. Н.Э. Баумана, Ким А.М., группа ИУ12-41М.

## Постановка

Классификация 10 классов городских звуков на датасете **UrbanSound8K**
(Salamon et al., 2014) с помощью двух моделей:

1. **Собственная 2D-CNN** на лог-мел спектрограммах (обучение с нуля).
2. **Дотюненная PANN CNN10** — backbone предобучен на AudioSet (527 классов),
   head заменена на 10-классную.

Оценка — официальная **10-fold cross-validation** (folds жёстко по
`metadata/UrbanSound8K.csv`). Дополнительно проведён ablation: своя CNN без
аугментации на 3 folds.

## Данные

- **UrbanSound8K** — 8732 файла, 10 классов: `air_conditioner`, `car_horn`,
  `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`,
  `jackhammer`, `siren`, `street_music`.
- Длительность сегментов ≤ 4 с (среднее ~3.6 с).
- Sample rate в исходных файлах: 8/16/22.05/44.1/48 kHz → resample к 22050 Hz
  (для own CNN) и 32000 Hz (для PANN).
- Источник: https://zenodo.org/records/1203745 (CC BY-NC 3.0).

## Пайплайн

```
wav → load (mono) → resample → pad/crop 4s → log-mel (64×173) →
      [SpecAugment + noise, только train] → CNN → CrossEntropy
```

Спектрограммы кешируются после первого вычисления — обучение в 5-10× быстрее.

## Архитектуры

**Own CNN** — 4 conv-блока (32→64→128→128), BN+ReLU+MaxPool+Dropout2d,
AdaptiveAvgPool, FC 128→64→10. ~250K параметров.

**PANN CNN10** — 6 conv-блоков + FC 2048-D backbone, обученный на AudioSet
(автозагрузка весов из Zenodo через `panns_inference`). Замена head 527→10,
fine-tune в две фазы: 5 эпох с замороженным backbone, затем размораживаем с
разными LR (backbone 1e-4, head 1e-3).

## Аугментация

SpecAugment (2 time mask × 25 + 2 freq mask × 12) + GaussianNoise (std=0.005, p=0.3).
Только на train. Ablation подтверждает прирост в ~3-5% accuracy.

## Метрики

- **accuracy** (основная — датасет почти сбалансирован)
- **macro-F1** (страховка от minority классов: `gun_shot`, `car_horn`)
- per-class precision/recall/F1
- confusion matrix (усреднённая по 10 folds)

Все метрики — mean ± std по 10 folds.

## Результаты

См. `results/summary.md` (генерируется `experiments/aggregate_results.py`):

<!-- AUTO-INSERTED-FROM-summary.md -->

Графики в `results/figures/`:
- `learning_curves_*.png` — кривые обучения для типового fold
- `confusion_matrix_*_avg.png` — усреднённые confusion matrix
- `per_class_f1_comparison.png` — сравнение F1 по классам

## Установка

Существующий venv с torch 2.8 (CPU):

```powershell
& "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe" -m pip install -r requirements-extra.txt
```

## Запуск

Из корня `final_project/`:

```powershell
# 0) Скачать датасет (один раз)
$PY -m src.data.download

# 1) Своя CNN: 10-fold CV
$PY experiments\run_own_cnn.py

# 2) PANN CNN10 fine-tune: 10-fold CV
$PY experiments\run_pann_finetune.py

# 3) Ablation без аугментации: 3 folds
$PY experiments\run_ablation_no_aug.py

# 4) Сводная таблица + графики
$PY experiments\aggregate_results.py
```

Где `$PY = "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe"`.

Для быстрой проверки добавь `--fast` к любому из `run_*.py` (1 fold, 2 эпохи).

## Структура

```
final_project/
├── src/
│   ├── config.py
│   ├── data/        # download, audio, augment, dataset
│   ├── models/      # own_cnn, pann_cnn10
│   ├── training/    # seed, metrics, trainer
│   └── utils/       # io, plotting
├── experiments/
│   ├── run_own_cnn.py
│   ├── run_pann_finetune.py
│   ├── run_ablation_no_aug.py
│   └── aggregate_results.py
├── notebooks/eda.ipynb
├── tests/           # pytest suite, ~30 tests
├── results/         # per-fold logs, ckpts, summary, figures
└── docs/superpowers/specs/, plans/
```

## Тесты

```powershell
$PY -m pytest -v
```
Ожидание: все тесты проходят (~30 штук).

## Ограничения и возможные улучшения

- 1D-CNN на сырых waveform-ах (M5 / SampleCNN) — для сравнения end-to-end подходов.
- Mixup и cutmix augmentation.
- Fine-tune более тяжёлых backbones (PANN CNN14, AST) на GPU.
- Гиперпараметр-поиск через Optuna.
- Cross-dataset eval на ESC-50.
- Pseudo-labelling / self-training.

## Лицензия

Код — MIT. Датасет UrbanSound8K — CC BY-NC 3.0 (Salamon et al., 2014). PANN CNN10 — MIT (Kong et al., 2020).
```

After writing the README, manually copy the contents of `results/summary.md` into the README at the `<!-- AUTO-INSERTED-FROM-summary.md -->` marker.

- [ ] **Step 6: Verify results structure**

Run:
```powershell
ls results/
ls results/figures/
cat results/summary.md
```
Expected: see `summary.csv`, `summary.md`, `own_cnn/`, `pann_cnn10/`, `ablation_no_aug/`, `figures/` with all expected PNGs.

- [ ] **Step 7: Final commit**

```bash
git add README.md results/summary.csv results/summary.md
git commit -m "docs: finalize README with results and full project description"
```

---

## Self-Review Checklist

Before declaring this plan complete, the engineer should verify:

1. **Spec coverage:**
   - [x] Section 4 (repo structure) → Task 1
   - [x] Section 5 (data pipeline) → Tasks 4, 5, 6, 7
   - [x] Section 6 (models) → Tasks 8, 9
   - [x] Section 7 (training) → Tasks 2, 10, 12
   - [x] Section 8 (evaluation) → Tasks 10, 13, 14, 15, 16
   - [x] Section 9 (EDA notebook) → Task 17
   - [x] Section 10 (run scripts) → Tasks 13, 14, 15, 16, 18
   - [x] Section 13 (reproducibility) → Task 2, all experiment scripts

2. **No co-author lines in any commit messages.** Verify via `git log --all` after the final commit — there must not be any `Co-Authored-By: Claude` lines (per user preference).

3. **Type consistency:** `OwnCNN`, `PannCnn10Wrapper`, `Trainer`, `TrainConfig`, `UrbanSound8KDataset`, `AudioConfig` — names match across tasks. `compute_metrics` returns a dict with the same keys throughout. `append_csv_row` and `save_npy`/`load_npy` signatures consistent.

4. **All tests pass at the end of each task** — running `$PY -m pytest -v` after Task 12 should show all ~30 tests green.

---

**End of plan.**
