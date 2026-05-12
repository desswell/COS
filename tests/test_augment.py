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
