# Multispec Rotated RT-DETR

8-channel multi-spectral extension of [O2-RTDETR](../rotated_rtdetr/README.md)
for HSMOT and other multi-spectral rotated detection tasks.

## Features

- `MultispecResNetV1dPaddle3DSE`: ResNetV1d with 3D+SE replacement for deep-stem
  first 3x3 conv (default backbone)
- `MultispecResNetV1dPaddle`: legacy 8-channel Conv2d stem with
  `rgbrepeat` / `interpolate` pretrain expansion
- HSMOT dataset integration via `HSMOTDataset`
- Compatible with the existing `RotatedRTDETR` detector

## Project Layout

```
multispec_rotated_rtdetr/
├── multispec_rotated_rtdetr/
│   ├── resnet.py              # MultispecResNetV1dPaddle3DSE / legacy backbone
│   ├── stem_conv3d_se.py      # 3D conv + pixel-wise SE stem module
│   └── pretrain_utils.py      # stem conv expansion utilities
├── configs/
│   ├── hsmot.py               # full HSMOT (3jpg, train_half)
│   ├── hsmot_debug.py         # debug: synthetic npy mini data
│   ├── hsmot_test.py          # test: real HSMOT mini splits
│   ├── pretrain_paths.py        # local O2-RTDETR checkpoint paths
│   ├── o2_rtdetr_r50vd_2xb4_72e_hsmot.py
│   ├── o2_rtdetr_r18vd_2xb4_72e_hsmot.py
│   ├── o2_rtdetr_r34vd_2xb4_72e_hsmot.py
│   ├── o2_rtdetr_r50vd_2xb4_72e_hsmot_pretrain.py
│   ├── o2_rtdetr_r18vd_2xb4_72e_hsmot_pretrain.py
│   ├── o2_rtdetr_r34vd_2xb4_72e_hsmot_pretrain.py
│   ├── o2_rtdetr_r18vd_2xb4_72e_hsmot_pretrain_ms.py
│   ├── o2_rtdetr_r18vd_1xb1_1e_hsmot_debug.py
│   └── o2_rtdetr_r18vd_1xb1_1e_hsmot_test.py
└── tools/
    ├── convert_pretrain_8ch.py
    ├── create_hsmot_debug_data.py
    ├── prepare_hsmot_test_splits.py
    ├── run_hsmot_debug_e2e.py
    └── run_hsmot_test_e2e.py
```

## Data Preparation

See [tools/data/hsmot/README.md](../../tools/data/hsmot/README.md).

Full dataset layout (3jpg under `PairMmot/data/hsmot/`):

```
data/hsmot/
├── train/npy2jpg/
├── test/npy2jpg/
└── train_half.txt
```

## Train (Full Dataset)

```bash
# R50, 2 GPUs, batch=2 per GPU (ImageNet backbone init only)
bash tools/dist_train.sh \
  projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_hsmot.py 2

# R18
bash tools/dist_train.sh \
  projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_hsmot.py 2
```

## Train with Local O2-RTDETR Pretrain

Place checkpoints under `PairMmot/pretrained_weights/` (paths in
`configs/pretrain_paths.py`). Use `*_pretrain.py` configs to load the full
O2-RTDETR detector (neck / encoder / decoder + backbone stem mapping):

```bash
# R50 from DOTA epoch_72
bash tools/dist_train.sh \
  projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_hsmot_pretrain.py 2

# R18 / R34
bash tools/dist_train.sh \
  projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_hsmot_pretrain.py 2
bash tools/dist_train.sh \
  projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_hsmot_pretrain.py 2
```

## Test (Full Dataset)

```bash
bash tools/dist_test.sh \
  projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_hsmot.py \
  work_dirs/o2_rtdetr_r50vd_2xb4_72e_hsmot/epoch_72.pth 2
```

## Debug Smoke Test (synthetic data)

Synthetic 2-frame mini dataset; no real HSMOT required:

```bash
python projects/multispec_rotated_rtdetr/tools/run_hsmot_debug_e2e.py

# isolated under PairMmot/tmp (recommended)
python projects/multispec_rotated_rtdetr/tools/run_hsmot_debug_e2e.py --tmpdir
# data -> PairMmot/tmp/hsmot_debug_e2e/HSMOT_mini
# ckpt -> PairMmot/tmp/hsmot_debug_e2e/work_dir/

# unit tests (py310, CUDA required for e2e)
python -m pytest tests/test_projects/test_multispec_rotated_rtdetr_hsmot_e2e.py -v
```

## Test Smoke Test (real HSMOT subset)

Real data at `PairMmot/data/hsmot/` (`train/npy2jpg`, `test/npy2jpg`).
Split lists and outputs stay **outside** the dataset tree:

```
PairMmot/
├── data/hsmot/              # read-only
├── tmp/hsmot_splits/        # test_mini.txt, train_mini.txt
└── tmp/hsmot_test_e2e/      # checkpoints & eval json (--use-tmp-workdir)
```

```bash
# 1-epoch integration test on 2 train + 2 test sequences
python projects/multispec_rotated_rtdetr/tools/run_hsmot_test_e2e.py \
  --use-tmp-workdir

# outputs in work_dirs/
python projects/multispec_rotated_rtdetr/tools/run_hsmot_test_e2e.py
```

## Convert Full Checkpoint to 8 Channels

If you already have a 3-channel O2-RTDETR checkpoint and want an 8-channel
version before fine-tuning:

```bash
python projects/multispec_rotated_rtdetr/tools/convert_pretrain_8ch.py \
  work_dirs/o2_rtdetr_r50vd_2xb4_72e_dota/epoch_72.pth \
  work_dirs/o2_rtdetr_r50vd_2xb4_72e_dota/epoch_72_8ch.pth \
  --expand-mode rgbrepeat
```

Then set `load_from` in the config to the converted checkpoint.

## Key Config Notes

| Item | Value |
|------|-------|
| Input channels | 8 |
| Classes (HSMOT) | 8 |
| Mean / Std | HSMOT 8-channel stats in config |
| Backbone | `MultispecResNetV1dPaddle3DSE(in_channels=8, num_spectral=8)` |
| Image loader | `LoadMultichannelImageFromNpy` or `LoadMultichannelImageFrom3JPG` |

Config naming:

| Suffix | Purpose |
|--------|---------|
| (none) | Full HSMOT training / eval |
| `_debug` | Synthetic mini data smoke test |
| `_test` | Real HSMOT subset integration test |

To switch image format in full config, edit `img_format` in `configs/hsmot.py`:

```python
img_format = 'npy'    # or '3jpg'
data_prefix=dict(img_path='npy')  # or 'npy2jpg'
```

The default backbone replaces ResNetV1d ``stem.0`` with a 3D spectral conv
(``kernel=(3,3,3)``) plus pixel-wise SE fusion. Pretrained ``stem.0`` Conv2d
weights map to ``conv3d``; SE layers start with uniform band weights
(``1/8`` per band).

To use the legacy 8-channel Conv2d stem with spectral interpolation:

```python
backbone=dict(
    type=MultispecResNetV1dPaddle,
    in_channels=8,
    expand_mode='interpolate',
    ...
)
```

## Citation

If you use O2-RTDETR, please cite the original paper listed in
[rotated_rtdetr/README.md](../rotated_rtdetr/README.md).
