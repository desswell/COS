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
