#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

conda run -n py310 python train.py \
  --dataset hsmot \
  --root_path ../data/hsmot/train \
  --ann_file ../train_half.txt \
  --ann_subdir mot \
  --img_subdir npy2jpg \
  --img_format 3jpg \
  --image_scale 800 1200 \
  --depth 18 \
  --epochs 100 \
  --batch_size 4 \
  --workers 4 \
  --device cuda:0 \
  --checkpoint_interval 1 \
  --model_dir ../workdir/ctracker_hsmot_r18_3dse_rotated
