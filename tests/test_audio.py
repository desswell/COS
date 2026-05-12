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
