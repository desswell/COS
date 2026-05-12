# Дизайн: классификация городских звуков на UrbanSound8K

**Курс:** «Методы искусственного интеллекта в ЦОС», МГТУ им. Н.Э. Баумана
**Студент:** Ким А.М., ИУ12-41М
**Дата:** 2026-05-11
**Расположение проекта:** `C:\Users\desswell\work\pythonProject\ЦОС\final_project`

## 1. Постановка задачи

Обучить нейронную сеть для классификации городских звуков (10 классов) на датасете UrbanSound8K. В рамках курса требуется:

- Анализ предметной области и EDA данных;
- Обоснованный выбор метрик;
- Обучение собственной модели (допускается использование pre-training);
- Полноценный ML-пайплайн с логами обучения;
- Оценка качества и разбор ошибок.

Дополнительно (для презентации): сравнение собственной модели с дотюненной предобученной моделью PANN CNN10 и ablation-эксперимент с отключённой аугментацией.

## 2. Данные

**Датасет:** UrbanSound8K (Salamon et al., 2014).
**Источник:** https://zenodo.org/records/1203745 (CC BY-NC 3.0).
**Размер:** 8732 аудиофайла, ~6 GB.
**Длительность:** до 4 секунд, среднее ~3.6 с.
**Sample rate:** разный (8–48 kHz) — резэмплинг на лету.
**Формат:** WAV.

**Классы (10):**
`air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`.

**Splits:** 10 предзаданных folds в `metadata/UrbanSound8K.csv` (колонка `fold`). Используются строго по официальному протоколу — без перемешивания файлов между folds (иначе data leakage из-за того, что фрагменты одного source recording могут попадать в разные folds).

**Дисбаланс:** датасет почти сбалансирован (по 800–1000 файлов на класс), кроме `gun_shot` (374) и `car_horn` (429). Class weights не применяются на старте — будут добавлены как «решение проблемы» при необходимости (по EDA / по результатам).

## 3. Подход

Сравниваются три эксперимента, оценка — официальная 10-fold cross-validation.

| Эксперимент | Модель | Назначение |
|---|---|---|
| **Own CNN** | Собственная 2D-CNN на лог-мел спектрограммах, обучение с нуля | Основная модель, «собственная модель» по ТЗ |
| **PANN CNN10 fine-tune** | Дотюненная PANN CNN10 (AudioSet 527 → 10 классов) | Pretrained baseline для сравнения, «верхняя планка» |
| **Own CNN, no-aug** | Та же CNN, аугментация отключена | Ablation на 3 folds, демонстрация эффекта аугментации |

## 4. Структура репозитория

```
final_project/
├── README.md
├── requirements-extra.txt        # доп. пакеты (panns_inference, soundfile, librosa)
├── data/
│   └── UrbanSound8K/             # скачивается скриптом (~6 GB)
│       ├── audio/foldN/*.wav
│       └── metadata/UrbanSound8K.csv
├── src/
│   ├── __init__.py
│   ├── config.py                 # все константы препроцессинга и тренировки
│   ├── data/
│   │   ├── download.py           # скачать и распаковать датасет
│   │   ├── dataset.py            # UrbanSound8KDataset
│   │   ├── audio.py              # load/resample/pad-crop, log-mel
│   │   └── augment.py            # SpecAugment + gaussian noise
│   ├── models/
│   │   ├── own_cnn.py            # 2D-CNN, 4 conv-блока
│   │   └── pann_cnn10.py         # обёртка PANN + замена head
│   ├── training/
│   │   ├── trainer.py            # обучающий цикл, early stopping, ckpt
│   │   ├── metrics.py            # accuracy, macro-F1, per-class, confusion
│   │   └── seed.py               # seed_everything
│   └── utils/
│       ├── io.py                 # сохранение/загрузка CSV, JSON
│       └── plotting.py           # learning curves, confusion matrix
├── experiments/
│   ├── run_own_cnn.py            # 10-fold CV, своя CNN
│   ├── run_pann_finetune.py      # 10-fold CV, PANN CNN10
│   ├── run_ablation_no_aug.py    # 3 folds, своя CNN без аугментации
│   └── aggregate_results.py      # сводная таблица mean±std
├── notebooks/
│   └── eda.ipynb                 # EDA + примеры спектрограмм
└── results/
    ├── own_cnn/                  # per-fold logs (CSV) + ckpts + cm.npy
    ├── pann_cnn10/
    ├── ablation_no_aug/
    ├── summary.csv
    ├── summary.md
    └── figures/                  # PNG для презентации
```

## 5. Data pipeline

### 5.1. Загрузка датасета (`src/data/download.py`)

- Качаем архив `UrbanSound8K.tar.gz` с Zenodo через `urllib.request` с progress bar (`tqdm`).
- Проверяем SHA-256.
- Распаковываем в `data/UrbanSound8K/`.
- Идемпотентно: если уже распакован — пропускаем.

### 5.2. Параметры аудио (`src/config.py`)

```python
# Для собственной CNN
SAMPLE_RATE = 22050
DURATION_S = 4.0
N_SAMPLES = 88200  # SR * DURATION
N_FFT = 1024       # ~46 ms
HOP_LENGTH = 512   # ~23 ms, 50% overlap
N_MELS = 64
F_MIN = 20
F_MAX = 11025
OUTPUT_SHAPE = (1, 64, 173)  # (C, mel, frames)

# Для PANN CNN10 (требует свои параметры)
PANN_SAMPLE_RATE = 32000
PANN_N_FFT = 1024
PANN_HOP_LENGTH = 320
PANN_N_MELS = 64
PANN_F_MIN = 50
PANN_F_MAX = 14000
```

### 5.3. Препроцессинг (`src/data/audio.py`)

```
torchaudio.load → mono (mean over channels)
                → resample (22050 для own CNN, 32000 для PANN)
                → pad zeros (right) или center-crop до N_SAMPLES
                → MelSpectrogram → log10(mel + 1e-10)
                → per-sample normalize: (x - mean) / (std + 1e-6)
```

### 5.4. Кеширование спектрограмм

При первом проходе любого скрипта обучения mel-спектрограммы сохраняются в:
- `data/UrbanSound8K/cache_mel/{fold}/{filename}.pt` — для own CNN (22 kHz)
- `data/UrbanSound8K/cache_mel_pann/{fold}/{filename}.pt` — для PANN (32 kHz)

На последующих эпохах и folds — `torch.load` без перевычисления. Ускорение CPU-обучения 5-10×.

### 5.5. Аугментация (`src/data/augment.py`), только train

```python
SpecAugment:
  time_masks:  2 × (max_width = 25 frames из 173)
  freq_masks:  2 × (max_width = 12 bins из 64)
GaussianNoise (на лог-мел спектрограмме):
  std = 0.005
  p = 0.3
```

Применяется в `__getitem__` после загрузки кеша. Аугментация — на спектрограмме, не на waveform, чтобы не ломать кеш и не замедлять DataLoader.

### 5.6. Dataset class (`src/data/dataset.py`)

```python
UrbanSound8KDataset(
    metadata_csv: Path,
    folds: list[int],         # какие folds брать
    train: bool,              # включает shuffle на уровне сэмплера
    augment: bool,            # включает аугментацию
    cache_dir: Path,          # cache_mel или cache_mel_pann
    audio_config: AudioConfig # own / pann параметры mel
)
```

### 5.7. DataLoader

- `batch_size = 32` (own CNN) / `16` (PANN, тяжелее)
- `num_workers = 0` (Windows + кеш делает параллелизм неактуальным)
- `shuffle = True` для train, `False` для val/test
- `pin_memory = False` (CPU)

## 6. Модели

### 6.1. Собственная 2D-CNN (`src/models/own_cnn.py`)

Архитектура (4 conv-блока, ~250K параметров — точное число фиксируется при реализации):

```
Input (1, 64, 173)
├─ ConvBlock1: Conv2d(1→32, k=3, p=1) → BN → ReLU → MaxPool(2,2) → Dropout2d(0.1)
│  out: (32, 32, 86)
├─ ConvBlock2: Conv2d(32→64, k=3, p=1) → BN → ReLU → MaxPool(2,2) → Dropout2d(0.15)
│  out: (64, 16, 43)
├─ ConvBlock3: Conv2d(64→128, k=3, p=1) → BN → ReLU → MaxPool(2,2) → Dropout2d(0.2)
│  out: (128, 8, 21)
├─ ConvBlock4: Conv2d(128→128, k=3, p=1) → BN → ReLU → MaxPool(2,2) → Dropout2d(0.25)
│  out: (128, 4, 10)
├─ AdaptiveAvgPool2d(1,1) → Flatten → (128,)
├─ Linear(128 → 64) → ReLU → Dropout(0.3)
└─ Linear(64 → 10)
```

Conv-веса инициализируются Kaiming-normal. BN — стандартный init.

### 6.2. PANN CNN10 fine-tune (`src/models/pann_cnn10.py`)

- Загружаем pretrained CNN10 через `panns_inference` (веса автоскачиваются).
- Заменяем `fc_audioset: Linear(2048 → 527)` на `Linear(2048 → 10)`, Kaiming init.
- Используем mel-параметры PANN (32 kHz, n_fft=1024, hop=320, n_mels=64, fmin=50, fmax=14000) — отдельный кеш.

**Стратегия fine-tune (две фазы):**

| Phase | Epochs | Trainable | LR |
|---|---|---|---|
| 1 (warmup) | 1–5 | only new head | 1e-3 |
| 2 (fine-tune) | 6–30 | all params | backbone 1e-4, head 1e-3 (parameter groups) |

## 7. Тренировка

**Trainer (`src/training/trainer.py`)** общий для обеих моделей:

- Optimizer: AdamW, `weight_decay=1e-4`
- Loss: CrossEntropy
- Scheduler: `CosineAnnealingLR(T_max=epochs)`
- Gradient clipping: `max_norm=1.0`
- Early stopping: по `val_acc`, patience указывается per-experiment
- Mixed precision: нет (CPU)
- Checkpoint: лучший по val_acc сохраняется в `fold_<i>_best.pt`

**Per-fold split:**
- `test = {fold_id}`
- `train_full = folds \ {fold_id}` → случайно делим 90% train / 10% val (seed=42, фиксированный для воспроизводимости)
- Метрика отчёта — на `test_fold`

**Гиперпараметры:**

| | Own CNN | PANN CNN10 |
|---|---|---|
| epochs | 50 | 30 |
| early stopping patience | 10 | 8 |
| batch size | 32 | 16 |
| LR | 1e-3 | head 1e-3, backbone 1e-4 (phase 2) |

**Seed:** `seed_everything(fold_id * 100 + 42)` в начале каждого fold (фиксирует numpy, torch, random, PYTHONHASHSEED).

## 8. Evaluation

### 8.1. Метрики (`src/training/metrics.py`)

**Per-fold:**
- `accuracy` (основная, датасет почти сбалансирован)
- `macro_f1` (страховка от minority classes)
- `per_class_precision`, `per_class_recall`, `per_class_f1`
- `confusion_matrix` (10×10) → `results/<model>/fold_<i>_cm.npy`
- `train_time_s`, `inference_time_ms_per_sample`, `best_epoch`, `n_params`

**Aggregate по 10 folds:**
- `accuracy_mean ± accuracy_std`, `f1_mean ± f1_std`
- усреднённая нормированная по строкам confusion matrix

### 8.2. Логи обучения

Per-fold CSV в `results/<model>/fold_<i>.csv`:
```
epoch, train_loss, train_acc, val_loss, val_acc, val_f1, lr, time_s
```

Учебный план для презентации — построить learning curves для 1 типичного fold (по умолчанию fold 0) и положить в `results/figures/`.

### 8.3. Сводный отчёт (`experiments/aggregate_results.py`)

Собирает per-fold CSV и пишет:

- `results/summary.csv` — одна строка на эксперимент:
  ```
  model, n_folds, acc_mean, acc_std, f1_mean, f1_std,
  train_time_s_total, n_params, best_epoch_avg
  ```
- `results/summary.md` — та же таблица в markdown, для вставки в README
- `results/figures/`:
  - `learning_curves_<model>_fold0.png`
  - `confusion_matrix_<model>_avg.png`
  - `per_class_f1_comparison.png` (own CNN vs PANN side-by-side bars)
  - `aug_effect.png` (спектрограмма до/после SpecAugment)

## 9. EDA notebook (`notebooks/eda.ipynb`)

Структура (для слайда «Анализ обучающего набора данных»):

1. Обзор датасета (описание, лицензия, ссылки, общая статистика).
2. Распределение классов (bar chart по `classID`).
3. Распределение классов по folds (heatmap 10×10).
4. Распределение длительности файлов (гистограмма, % требующих pad / crop).
5. Распределение sample rate-ов исходных файлов (мотивация для resample к 22050).
6. Примеры waveform-ов — по 1 на класс (`librosa.display.waveshow`).
7. Примеры log-mel спектрограмм — по 1 на класс (`librosa.display.specshow`).
8. Эффект аугментации — спектрограмма до / SpecAugment / +noise.
9. Выводы EDA — наблюдения и решения, заложенные в препроцессинг.

## 10. Запуск

```powershell
# 0) Установка доп. пакетов
& "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe" -m pip install -r requirements-extra.txt

# 1) Скачать датасет (~6 GB, делается один раз)
& "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe" -m src.data.download

# 2) Основная модель: 10-fold CV
& "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe" experiments\run_own_cnn.py

# 3) PANN fine-tune baseline: 10-fold CV
& "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe" experiments\run_pann_finetune.py

# 4) Ablation без аугментации: 3 folds
& "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe" experiments\run_ablation_no_aug.py

# 5) Сводная таблица + графики
& "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe" experiments\aggregate_results.py
```

## 11. Бюджет времени (CPU)

| Эксперимент | Folds | Ожидаемое время |
|---|---|---|
| Кеш mel при первом проходе | — | ~30 мин |
| Own CNN | 10 | ~2 ч |
| Own CNN no-aug ablation | 3 | ~30-45 мин |
| PANN CNN10 fine-tune | 10 | ~4 ч |
| **Итого** | | **~7 ч** (на ночь) |

## 12. Зависимости поверх существующего venv

Существующий venv `C:\Users\desswell\work\pythonProject\.venv` уже содержит: torch 2.8 (CPU), torchaudio, numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm, einops, datasets, transformers.

Дополнительно нужно установить (`requirements-extra.txt`):
- `panns_inference` (PANN CNN10 + автоскачивание весов)
- `soundfile` (бэкенд для torchaudio на Windows для wav)
- `librosa` (для EDA: duration analysis, display)

## 13. Воспроизводимость

- Все random seeds фиксированы (`seed_everything(fold_id * 100 + 42)`).
- 90/10 train/val split детерминирован (seed=42).
- Конфиги в `src/config.py`, не разбросаны по скриптам.
- Per-fold CSV и checkpoints коммитятся в `results/` (исключая большие веса, см. .gitignore).

## 14. Что НЕ делаем (out of scope)

- 1D-CNN на raw waveform (упомянуть в «возможные улучшения»).
- Mixup augmentation (упомянуть в «возможные улучшения»).
- Fine-tune PANN CNN14 / AST (слишком медленно на CPU).
- Cross-dataset eval на ESC-50.
- Гиперпараметр-поиск через Optuna.
- Auto-push на GitHub — репо ведётся локально.
- Co-authoring строки в git-коммитах — не добавляем.

## 15. Соответствие ТЗ

| Требование ТЗ | Где закрыто |
|---|---|
| Исследование предметной области | README (раздел «контекст»), презентация |
| EDA | `notebooks/eda.ipynb` |
| Выбор метрик | accuracy + macro-F1 + per-class + confusion |
| Обучение собственной модели | `src/models/own_cnn.py`, `experiments/run_own_cnn.py` |
| Pre-training (опционально) | `src/models/pann_cnn10.py`, `experiments/run_pann_finetune.py` |
| Логи обучения | `results/<model>/fold_<i>.csv` + learning curves |
| Оценка качества | `results/summary.csv` + `summary.md` + confusion matrices |
| ML-пайплайн | download → cache → train → eval → aggregate |
| Код на GitHub | локальный git-репо, готовый к `git push` |
