#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
CONDA_BIN="${CONDA_BIN:-/data1/users/litianhao01/anaconda3/bin/conda}"

# RTX 5090 single-GPU profile: micro-batch 4 x accumulation 2 preserves the
# original CTracker effective batch size of 8.
"${CONDA_BIN}" run --no-capture-output -n py310 python -u train.py \
  --dataset hsmot \
  --root_path ../data/hsmot/train \
  --ann_subdir mot \
  --img_subdir npy2jpg \
  --img_format 3jpg \
  --image_scale 900 1200 \
  --depth 50 \
  --epochs 100 \
  --batch_size 4 \
  --accumulation_steps 2 \
  --workers 32 \
  --device cuda:0 \
  --pretrained_model /data4/litianhao/PairMmot/pretrained_weights/ctracker_model_final.pt \
  --lr 5e-5 \
  --stem_lr_multiplier 10 \
  --checkpoint_interval 1 \
  --model_dir /data4/litianhao/PairMmot/workdir_178/ctracker_hsmot_r50_3dse_rotated_1200x900_bs4_acc2
