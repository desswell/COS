"""PANN CNN10 wrapper for fine-tuning on UrbanSound8K.

The installed `panns_inference` package ships only `Cnn14`, not `Cnn10`.
We therefore define a minimal `Cnn10` backbone here that exactly mirrors the
upstream layer naming from
https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
so that the official AudioSet `Cnn10_mAP=0.380.pth` checkpoint loads with
`load_state_dict(..., strict=False)` (we drop the `spectrogram_extractor` /
`logmel_extractor` / `spec_augmenter` keys, which we do not need).

The forward signature accepts pre-computed log-mel spectrograms (so we can
reuse the spectrogram cache), not raw waveforms — we bypass PANN's internal
stft/mel layers.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

# Reuse the official ConvBlock + init helpers from the installed package so
# our state-dict keys are byte-identical to the released checkpoint.
from panns_inference.models import ConvBlock, init_bn, init_layer


class Cnn10(nn.Module):
    """Backbone matching the upstream PANN Cnn10 architecture (4 conv blocks).

    We omit the `spectrogram_extractor`, `logmel_extractor`, and `spec_augmenter`
    sub-modules of the upstream model — our wrapper feeds pre-computed log-mel
    spectrograms directly. The layer names that DO exist (`bn0`, `conv_block1..4`,
    `fc1`, `fc_audioset`) match the upstream naming so the pretrained checkpoint
    loads via `strict=False`.
    """

    def __init__(self, classes_num: int = 527, mel_bins: int = 64):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self.init_weight()

    def init_weight(self) -> None:
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)


class PannCnn10Wrapper(nn.Module):
    """CNN10 backbone + new linear classifier (replaces the 527-way AudioSet head).

    When `pretrained=True`, downloads & loads AudioSet weights from Zenodo.
    When False, initialises randomly (used for unit tests).
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True, mel_bins: int = 64):
        super().__init__()

        backbone = Cnn10(classes_num=527, mel_bins=mel_bins)

        if pretrained:
            ckpt_path = _download_cnn10_checkpoint()
            state = torch.load(ckpt_path, map_location="cpu")
            sd = state.get("model", state)
            # Drop keys that belong to the stft/mel/spec_augmenter sub-modules
            # we don't carry — they would otherwise show up as "unexpected".
            sd = {
                k: v
                for k, v in sd.items()
                if not k.startswith(("spectrogram_extractor", "logmel_extractor", "spec_augmenter"))
            }
            missing, unexpected = backbone.load_state_dict(sd, strict=False)
            if unexpected:
                print(f"[pann] {len(unexpected)} unexpected keys (ignored): {unexpected[:3]}...")
            if missing:
                non_head_missing = [k for k in missing if not k.startswith("fc_audioset")]
                if non_head_missing:
                    print(f"[pann] {len(non_head_missing)} missing keys: {non_head_missing[:3]}...")

        # Replace 527-class head with `num_classes` head. Keep the 512-dim fc1 features.
        in_features = backbone.fc_audioset.in_features
        backbone.fc_audioset = nn.Identity()  # bypass original head, take pre-head features
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity="relu")
        nn.init.zeros_(self.classifier.bias)

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """log_mel: [B, 1, n_mels, T] — same layout as our dataset.

        Replicates the upstream Cnn10.forward() body, but skips the
        `spectrogram_extractor` / `logmel_extractor` steps. The upstream model
        keeps the spec in layout [B, 1, T, F]; our dataset uses [B, 1, F, T],
        so we transpose first.
        """
        x = log_mel.transpose(2, 3)                          # [B, 1, F, T] -> [B, 1, T, F]
        x = x.transpose(1, 3)                                # -> [B, F, T, 1]
        x = self.backbone.bn0(x)                             # batchnorm over freq axis
        x = x.transpose(1, 3)                                # -> [B, 1, T, F]

        x = self.backbone.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.backbone.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.backbone.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.backbone.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)                             # mean over freq
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.backbone.fc1(x))
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

    `panns_inference` doesn't ship CNN10 (only CNN14). We pull CNN10 directly
    from the official Zenodo mirror.
    """
    cache = Path(os.path.expanduser("~/.cache/panns_inference"))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "Cnn10_mAP=0.380.pth"
    if not path.exists():
        url = "https://zenodo.org/records/3987831/files/Cnn10_mAP%3D0.380.pth"
        print(f"[pann] Downloading CNN10 weights to {path} ...")
        urllib.request.urlretrieve(url, str(path))
    return str(path)
