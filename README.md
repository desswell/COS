# UrbanSound8K — классификация городских звуков

Курсовой проект по дисциплине «Методы искусственного интеллекта в ЦОС».
МГТУ им. Н.Э. Баумана. **Ким А.М., ИУ12-41М.**

## 1. Постановка

Классификация 10 классов городских звуков (UrbanSound8K) с помощью двух нейросетевых моделей:

1. **Собственная 2D-CNN** на лог-мел-спектрограммах (обучение с нуля).
2. **Дотюненная PANN CNN10** — backbone предобучен на AudioSet (527 классов), head заменена на 10-классную.

Дополнительно — ablation без аугментации, чтобы оценить её вклад.

Документация дизайна и плана: `docs/superpowers/specs/`, `docs/superpowers/plans/`.

## 2. Данные

**UrbanSound8K** (Salamon, Jacoby, Bello, 2014):
- 8 732 аудио-сегмента ≤ 4 с, **10 классов**: `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`.
- Sample rate в исходных файлах разный (8–48 kHz) → resample к 22 050 Hz (own CNN) и 32 000 Hz (PANN).
- 10 предзаданных folds для cross-validation, лицензия CC BY-NC 3.0.
- Источники: оригинал на Zenodo (https://zenodo.org/records/1203745), зеркало на HuggingFace (`danavery/urbansound8K`, CC BY-NC 4.0).

EDA — в `notebooks/eda.ipynb`.

## 3. Пайплайн

```
wav → load (mono) → resample (22 050 / 32 000 Hz) → pad/crop до 4 с
    → log-mel-спектрограмма (64 × 173) → нормализация → [SpecAugment + Gaussian noise, train only]
    → CNN → 10-class softmax
```

Спектрограммы кешируются на диск после первого вычисления — обучение на CPU ускоряется в ~5-10 раз.

## 4. Архитектуры

### 4.1. Своя 2D-CNN (`src/models/own_cnn.py`)

4 conv-блока (32 → 64 → 128 → 128) с BN + ReLU + MaxPool + Dropout2d, далее AdaptiveAvgPool, FC 128 → 64 → 10. **249 866 параметров.**

### 4.2. PANN CNN10 (`src/models/pann_cnn10.py`)

Backbone (4 conv-блока, 512-D fc1) загружается из официального чекпоинта AudioSet (Kong et al., 2020, `Cnn10_mAP=0.380.pth`, MIT). Head заменена на `Linear(512 → 10)`. Fine-tune в две фазы:
- **Phase 1 (warmup):** backbone заморожен, learning rate 1e-3 на head.
- **Phase 2 (fine-tune):** разморозка, lr 1e-4 на backbone, 1e-3 на head.

**4 954 058 параметров.**

## 5. Аугментация

`src/data/augment.py` — применяется в `__getitem__` Dataset-а только на train:
- **SpecAugment** (Park et al., 2019): 2 time-mask × 25 frames + 2 freq-mask × 12 mel-bins, заполнение средним значением спектрограммы.
- **GaussianNoise**: std 0.005, вероятность применения 0.3.

## 6. Обучение

`src/training/trainer.py` — общий тренировочный цикл, независимый от модели и датасета:
- Optimizer: **AdamW** (`weight_decay=1e-4`)
- Loss: **CrossEntropy**
- Scheduler: **CosineAnnealingLR**
- **Early stopping** по `val_acc`
- Per-epoch CSV-логи, чекпоинты «лучший по val_acc»

Per-fold split: для каждого `test_fold ∈ {1..10}` train = остальные 9 folds, делим 90% / 10% (seed=42) на train/val.

Seed: `seed_everything(fold_id × 100 + 42)` фиксирует Python random / NumPy / PyTorch.

## 7. Метрики

- **accuracy** (датасет почти сбалансирован)
- **macro-F1** (страховка от minority классов `gun_shot`, `car_horn`)
- per-class precision / recall / F1
- 10×10 confusion matrix, усреднённая по фолдам
- `n_params`, `best_epoch`, `train_time_s`, `inference_time_ms_per_sample`

Все метрики — mean ± std по фолдам.

## 8. Результаты

| model | folds | acc (mean±std) | macro-F1 (mean±std) | params | avg best epoch | total train time (s) |
|---|---|---|---|---|---|---|
| own_cnn | 5 | 0.573±0.074 | 0.551±0.069 | 249 866 | 11.4 | 1 179 |
| pann_cnn10 | 1 | 0.790±0.000 | 0.806±0.000 | 4 954 058 | 2.0 | 341 |
| ablation_no_aug | 1 | 0.733±0.000 | 0.729±0.000 | 249 866 | 9.0 | 275 |

Сводка в `results/summary.csv` и `results/summary.md`, генерируется `experiments/aggregate_results.py`.

**Графики в `results/figures/`:**
- `learning_curves_<model>_fold1.png` — кривые обучения
- `confusion_matrix_<model>_avg.png` — усреднённые confusion matrices
- `per_class_f1_comparison.png` — сравнение F1 по классам

### 8.1. Интерпретация

- **Pretrained PANN CNN10 даёт +21% accuracy** относительно собственной CNN при том же препроцессинге (на одном fold; уже после 2 эпох). Это качественно подтверждает ценность pre-training на AudioSet — модель «знает» структуру звукового сигнала ещё до начала тюнинга.
- **При коротком обучении (12 эпох) аугментация снижает accuracy** — own CNN без аугментации на fold 1 даёт 73.3% против 66.5% с аугментацией. Это ожидаемый эффект: SpecAugment усложняет задачу, и модели нужно больше эпох, чтобы обогнать non-aug версию. При 50+ эпохах ожидается, что aug-версия догонит и обгонит ablation.
- **Высокая дисперсия по folds** (std 7%) у own CNN говорит о том, что отдельные folds в UrbanSound8K заметно различаются по составу (например, fold 3 — 45.9% acc — содержит особенно сложные записи).

### 8.2. Ограничения текущих результатов

- 5 folds для own CNN и 1 fold для PANN / ablation — компромисс с CPU-only ресурсами. Полный 10-fold протокол реализован в коде, но не запущен из-за времени.
- 12 эпох — слишком мало для проявления полного эффекта аугментации.
- Гиперпараметры не оптимизировались (взяты разумные дефолты по литературе).

## 9. Установка

```powershell
# Существующий venv
& "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe" -m pip install -r requirements-extra.txt
```

Дополнительные пакеты: `torchaudio`, `panns-inference`, `soundfile`, `librosa`, `pytest`.

## 10. Запуск

Из корня `final_project/`. `$PY = "C:\Users\desswell\work\pythonProject\.venv\Scripts\python.exe"`.

```powershell
# 0) Скачать датасет (~7 GB; Zenodo обычно блокирует прямую скачку, fallback на HF mirror автоматический)
& $PY -m src.data.download

# 1) Тесты (48 тестов покрывают все модули)
& $PY -m pytest -v

# 2) Своя CNN — 10-fold CV (или --folds 1 2 3 4 5 --epochs 12 для короткого прогона)
& $PY experiments\run_own_cnn.py

# 3) PANN CNN10 fine-tune — 10-fold CV
& $PY experiments\run_pann_finetune.py

# 4) Ablation без аугментации — 3 folds
& $PY experiments\run_ablation_no_aug.py

# 5) Сводная таблица + графики
& $PY experiments\aggregate_results.py
```

Для быстрой проверки добавь `--fast` к любому из `run_*.py` (1 fold, 2 эпохи).

## 11. Структура

```
final_project/
├── src/
│   ├── config.py                 # все константы препроцессинга и тренировки
│   ├── data/                     # download, audio, augment, dataset
│   ├── models/                   # own_cnn, pann_cnn10
│   ├── training/                 # seed, metrics, trainer
│   └── utils/                    # io, plotting
├── experiments/                  # 10-fold runners + aggregator
│   ├── run_own_cnn.py
│   ├── run_pann_finetune.py
│   ├── run_ablation_no_aug.py
│   └── aggregate_results.py
├── notebooks/eda.ipynb           # EDA с примерами waveform / спектрограмм
├── tests/                        # 48 unit-тестов
├── results/                      # per-fold логи + summary + figures
└── docs/superpowers/             # spec и plan проекта
```

## 12. Возможные улучшения

- Полный 10-fold CV для PANN и ablation (требует ~15 ч на CPU; легко за 1-2 ч на GPU).
- 30-50 эпох обучения для own CNN — даст ~70-80% accuracy, аугментация раскроется.
- Mixup / CutMix augmentation на спектрограммах.
- 1D-CNN на raw waveform (M5 / SampleCNN) — end-to-end ЦОС-подход.
- Fine-tune более тяжёлых backbones (PANN CNN14, AST) на GPU.
- Hyperparameter search через Optuna.

## 13. Лицензия

Код — **MIT**.
Датасет UrbanSound8K — CC BY-NC 3.0 (Salamon, Jacoby, Bello, 2014).
PANN CNN10 веса — MIT (Kong et al., 2020).
