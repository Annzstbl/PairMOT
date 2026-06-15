# Data Preparation for HSMOT

HSMOT is a rotated-box multi-spectral MOT dataset with 8 channels.
It is recommended to symlink the dataset root to `$ai4rs/data/HSMOT`.

## Directory Layout

```
HSMOT/
├── train/
│   ├── mot/                         # sequence-level MOT annotations
│   │   ├── data23-1.txt
│   │   └── ...
│   ├── npy/                         # 8-channel frames (recommended)
│   │   ├── data23-1/
│   │   │   ├── 000001.npy
│   │   │   └── ...
│   │   └── ...
│   ├── npy2jpg/                     # optional 3-JPG storage
│   │   ├── data23-1/
│   │   │   ├── 000001_p1.jpg
│   │   │   ├── 000001_p2.jpg
│   │   │   ├── 000001_p3.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ImageSets/
│       ├── train.txt                # one sequence name per line
│       └── val.txt
└── test/
    ├── mot/
    ├── npy/
    └── ImageSets/
        └── test.txt
```

## Annotation Format

Each line in `mot/<seq>.txt` contains 13 comma-separated fields:

```
frame_id, track_id, x1, y1, x2, y2, x3, y3, x4, y4, score, class_id, truncation
```

- `frame_id`: 1-based frame index
- `track_id`: object ID within the sequence
- `x1..y4`: rotated quadrilateral in pixel coordinates
- `score`: GT is usually `-1`
- `class_id`: one of 8 classes (0-7)
- `truncation`: occlusion / truncation flag

## Classes

| ID | Name |
|----|------|
| 0 | car |
| 1 | bike |
| 2 | pedestrian |
| 3 | van |
| 4 | truck |
| 5 | bus |
| 6 | tricycle |
| 7 | awning-bike |

## Dataset Usage

### Detection + Track ID (MOT)

```python
dataset=dict(
    type='HSMOTDataset',
    data_root='data/HSMOT/train/',
    ann_subdir='mot',
    ann_file='ImageSets/train.txt',
    data_prefix=dict(img_path='npy'),
    img_format='npy',          # or '3jpg'
    with_track_id=True,
    pipeline=train_pipeline,
)
```

### Detection Only

Set `with_track_id=False` and use `HSMOTLoadAnnotations(with_track_id=False)`.

### 3-JPG Format

```python
data_prefix=dict(img_path='npy2jpg'),
img_format='3jpg',
```

## Config Reference

See `configs/_base_/datasets/hsmot.py` for a full dataloader example.

## Train / Test

```bash
# single GPU
python tools/train.py configs/_base_/datasets/hsmot.py

# multi GPU
bash tools/dist_train.sh <your_config_with_hsmot_base.py> 2
```

Note: the base config only defines the dataset. You still need a model config
that sets 8-channel input (backbone first conv + `DetDataPreprocessor` mean/std).
