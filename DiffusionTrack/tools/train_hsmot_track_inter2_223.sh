#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/linxu/code/DiffusionTrack-PairMOT
ENV=/data/users/linxu/.conda/envs/py310_pairmot
RUNTIME=/data4/linxu/PairMOT_DiffusionTrack
STAGE1=${STAGE1_CKPT:-${RUNTIME}/work_dirs/yolo11l_diffusion_det_hsmot_b4_d2_acc4_bf16_w8/best_ckpt.pth.tar}

export PATH="${ENV}/bin:/data/users/linxu/anaconda3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HSMOT_TRAIN_ROOT="${RUNTIME}/data/hsmot/train"
export HSMOT_VAL_ROOT="${RUNTIME}/data/hsmot/test"

cd "${REPO}"
exec python tools/train.py \
  -f exps/example/mot/yolo11l_diffusion_track_hsmot_inter2.py \
  -expn "${EXPERIMENT_NAME:-yolo11l_diffusion_track_hsmot_inter2_b4_d2_acc4_bf16_w8}" \
  -d 2 -b 4 --accumulate 4 --amp-dtype bf16 -c "${STAGE1}" \
  output_dir "${RUNTIME}/work_dirs" data_num_workers 8
