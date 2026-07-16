#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

conda run -n py310 python train.py \
  --dataset hsmot \
  --root_path ../data/hsmot/train \
  --ann_subdir mot \
  --img_subdir npy2jpg \
  --img_format 3jpg \
  --image_scale 900 1200 \
  --depth 50 \
  --epochs 100 \
  --batch_size 8 \
  --workers 32 \
  --device cuda:0 \
  --data_parallel \
  --pretrained_model /data4/litianhao/PairMmot/pretrained_weights/ctracker_model_final.pt \
  --lr 5e-5 \
  --stem_lr_multiplier 10 \
  --checkpoint_interval 1 \
  --model_dir ../workdir/ctracker_hsmot_r50_3dse_rotated_1200x900
