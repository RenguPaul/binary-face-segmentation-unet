# Binary Face Segmentation with U-Net

## About

Проект по бинарной семантической сегментации изображений лиц с использованием PyTorch и архитектуры U-Net.

Цель проекта — реализовать полный pipeline подготовки данных и обучения модели для задачи pixel-level segmentation: от загрузки и проверки пар `image / mask` до обучения U-Net, оценки качества сегментации и сохранения лучшего состояния модели.

Вся реализация находится в `My_UNet.ipynb`.

## What was implemented

В ноутбуке реализованы следующие этапы:

1. Конфигурация проекта через `dataclass Config` с параметрами датасета, preprocessing, обучения и воспроизводимости.

2. Автоматическое определение корня проекта для локального запуска и Google Colab.

3. Распаковка `archive.zip` и обработка двух вариантов структуры архива.

4. Загрузка датасета из пар `JPG + PNG` и проверка соответствия изображений и масок по имени файла.

5. Реализация `SegmentationDataset` для загрузки изображений и бинарных масок.

6. Предобработка изображений и масок:
   - изменение размера до `256 × 256`;
   - нормализация изображений;
   - преобразование в PyTorch Tensor;
   - бинаризация масок.

7. Аугментация обучающих данных с использованием Albumentations:
   - `HorizontalFlip`;
   - `VerticalFlip`;
   - `ShiftScaleRotate`;
   - `RandomBrightnessContrast`;
   - `HueSaturationValue`;
   - `CoarseDropout`.

8. Разбиение датасета на train, validation и test выборки с фиксированным seed.

9. Проверка подготовленных данных:
   - количество изображений и масок;
   - соответствие пар;
   - уникальные значения масок;
   - диапазон значений изображений;
   - визуализация примеров;
   - проверка формы batch.

10. Реализация архитектуры U-Net на PyTorch с encoder, decoder и skip connections.

11. Настройка функции потерь `Binary Cross Entropy + Dice Loss`.

12. Настройка метрики `Intersection over Union (IoU)`.

13. Настройка оптимизатора Adam и планировщика learning rate `CosineAnnealingLR`.

14. Реализация цикла обучения с расчётом loss и IoU на train и validation выборках.

15. Сохранение лучшего состояния модели в checkpoint.

16. Визуализация процесса обучения и результатов сегментации.

## Tech stack

- Python 3.10
- PyTorch 2.0+
- Albumentations
- NumPy
- Pillow
- Matplotlib
- Jupyter Notebook

## Dataset

Используется датасет из **974 пар изображение/маска**.

Каждая пара состоит из:

- `JPG` — RGB-изображение;
- `PNG` — бинарная маска;
- исходный размер изображения — `500 × 350`;
- маска хранится в градациях серого со значениями `{0, 255}`.

Изображение и маска сопоставляются по имени файла (`stem`).

Пример:

```text
000001.jpg
000001.png
```

Перед использованием маска бинаризуется:

```python
mask = (mask > 127).astype(np.float32)
```

После преобразования значения маски:

```text
{0, 1}
```

## Data split

Датасет разделяется с фиксированным seed `42`.

| Подмножество | Количество |
|---|---:|
| Train | 731 |
| Validation | 146 |
| Test | 97 |
| Всего | 974 |

Разбиение выполняется через `torch.utils.data.random_split`.

Для train и evaluation используются разные наборы трансформаций.

## Preprocessing

Все изображения и маски приводятся к размеру:

```text
256 × 256
```

Для обучающей выборки используются аугментации:

- `HorizontalFlip(p=0.5)`;
- `VerticalFlip(p=0.1)`;
- `ShiftScaleRotate` с параметрами `shift_limit=0.05`, `scale_limit=0.10`, `rotate_limit=15`, `border_mode=0`, `p=0.4`;
- `RandomBrightnessContrast` с `brightness_limit=0.15`, `contrast_limit=0.15`, `p=0.4`;
- `HueSaturationValue` с `hue_shift_limit=10`, `sat_shift_limit=15`, `val_shift_limit=10`, `p=0.3`;
- `CoarseDropout` с `1–4` отверстиями размером `8–24` px, `p=0.2`.

Геометрические преобразования применяются синхронно к изображению и маске.

Цветовые преобразования применяются только к изображению.

После preprocessing изображение нормализуется:

```text
mean = 0.5
std  = 0.5
```

Диапазон значений изображения после нормализации:

```text
[-1, 1]
```

Для преобразования в PyTorch Tensor используется `ToTensorV2`.

## Dataset

**Источник:** [Face Segmentation Dataset — Kaggle](https://www.kaggle.com/datasets/bemorekgg/face-segmentation-dataset)

Для загрузки данных реализован класс `SegmentationDataset`.

Один элемент датасета имеет форму:

```text
image: [3, 256, 256]
mask:  [1, 256, 256]
```

При `batch_size = 8`:

```text
images: [8, 3, 256, 256]
masks:  [8, 1, 256, 256]
```

## Data validation

Перед обучением выполняются проверки подготовленного датасета:

- количество JPG-файлов равно количеству PNG-файлов;
- каждому изображению соответствует маска с тем же `stem`;
- маски содержат только ожидаемые значения;
- после бинаризации значения масок находятся в `{0, 1}`;
- значения изображений находятся в диапазоне `[-1, 1]`;
- формы отдельных элементов и batch соответствуют ожидаемым;
- визуально проверяются несколько пар изображения и маски.

## Model

Для бинарной сегментации реализована архитектура U-Net на PyTorch.

Модель состоит из encoder и decoder частей со skip connections между соответствующими уровнями.

Вход:

```text
[batch, 3, 256, 256]
```

Выход:

```text
[batch, 1, 256, 256]
```

Количество обучаемых параметров модели:

```text
7,702,977
```

Выход модели используется для получения бинарной карты сегментации.

## Loss function

Для обучения используется комбинация:

```text
Binary Cross Entropy + Dice Loss
```

Комбинированная функция потерь используется для оптимизации качества бинарной сегментации.

## Evaluation metric

Для оценки качества сегментации используется **Intersection over Union (IoU)**.

IoU рассчитывается между предсказанной бинарной маской и ground truth маской.

```text
IoU = Intersection / Union
```

## Training

Основные параметры обучения:

```text
batch size     = 8
epochs         = 50
learning rate  = 1e-3
weight decay   = 1e-4
seed           = 42
```

В качестве оптимизатора используется:

```text
Adam
```

Для изменения learning rate используется:

```text
CosineAnnealingLR
```

Во время обучения рассчитываются значения loss и IoU для обучающей и validation выборок.

Лучшее состояние модели сохраняется в:

```text
checkpoints/best_unet.pth
```

## Reproducibility

В проекте используется фиксированный seed:

```text
42
```

Seed устанавливается для:

- Python `random`;
- NumPy;
- PyTorch;
- CUDA.

Также включается детерминированный режим cuDNN.

Основные параметры проекта собраны в `dataclass Config`.

Конфигурация включает:

- размер изображения;
- batch size;
- доли validation и test;
- количество workers;
- использование augmentation;
- количество эпох;
- learning rate;
- weight decay;
- seed;
- имя checkpoint.

## Project structure

```text
binary-face-segmentation-unet/
│
├── .gitignore
├── README.md
├── requirements.txt
├── My_UNet.ipynb
│
├── data/
│   └── .gitkeep
│
├── checkpoints/
│   └── .gitkeep
│
└── results/
    └── .gitkeep
```

В Git коммитятся только исходные файлы проекта и `.gitkeep`.

Датасет, архив, checkpoints и результаты выполнения ноутбука не хранятся в Git.

## Dataset setup

Ноутбук ожидает архив:

```text
archive.zip
```

в корне проекта.

Локальная структура перед запуском:

```text
binary-face-segmentation-unet/
│
├── archive.zip
├── My_UNet.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
├── checkpoints/
└── results/
```

После распаковки данные размещаются в:

```text
data/
└── segment/
```

Ноутбук поддерживает два варианта структуры архива.

### Вариант 1

Архив содержит директорию `segment`:

```text
archive.zip
└── segment/
    ├── image_001.jpg
    ├── image_001.png
    ├── image_002.jpg
    ├── image_002.png
    └── ...
```

### Вариант 2

Файлы находятся непосредственно в корне архива:

```text
archive.zip
├── image_001.jpg
├── image_001.png
├── image_002.jpg
├── image_002.png
└── ...
```

## Google Colab

При запуске в Google Colab архив размещается в:

```text
/content/archive.zip
```

Корень проекта определяется автоматически.

Рабочие директории:

```text
/content/data/
/content/checkpoints/
/content/results/
```

Структура после запуска:

```text
/content/
│
├── archive.zip
├── My_UNet.ipynb
│
├── data/
│   └── segment/
│
├── checkpoints/
└── results/
```

## Installation

Установка зависимостей:

```bash
pip install -r requirements.txt
```

После установки зависимостей открыть:

```text
My_UNet.ipynb
```

и выполнить ячейки ноутбука последовательно.
