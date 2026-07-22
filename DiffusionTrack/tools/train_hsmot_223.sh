#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/linxu/code/DiffusionTrack-PairMOT
ENV=/data/users/linxu/.conda/envs/py310_pairmot
RUNTIME=/data4/linxu/PairMOT_DiffusionTrack

export PATH="${ENV}/bin:/data/users/linxu/anaconda3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HSMOT_TRAIN_ROOT="${HSMOT_TRAIN_ROOT:-/data/users/qinhaolin01/data/hsmot/train}"
export HSMOT_VAL_ROOT="${HSMOT_VAL_ROOT:-/data/users/qinhaolin01/data/hsmot/test}"
export HSMOT_IMG_SUBDIR="${HSMOT_IMG_SUBDIR:-npy2jpg}"
export HSMOT_IMG_FORMAT="${HSMOT_IMG_FORMAT:-3jpg}"
export YOLO11_WEIGHTS="${YOLO11_WEIGHTS:-${RUNTIME}/pretrained_weights/mmot_official/yolo11L-8ch-3dstem.pt}"

cd "${REPO}"
train_args=(
  -f exps/example/mot/yolo11l_diffusion_det_hsmot.py
  -expn "${EXPERIMENT_NAME:-yolo11l_diffusion_det_hsmot_b4_d2_acc4_bf16_w8}"
  -d 2 -b 4 --accumulate 4 --amp-dtype bf16
)
if [[ "${RESUME:-0}" == "1" ]]; then
  train_args+=(--resume)
  if [[ -n "${CKPT:-}" ]]; then
    train_args+=(-c "${CKPT}")
  fi
fi
train_args+=(output_dir "${RUNTIME}/work_dirs" data_num_workers 8)
exec python tools/train.py "${train_args[@]}"
