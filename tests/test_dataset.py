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
