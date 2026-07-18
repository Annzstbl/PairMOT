#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/linxu/code/DiffusionTrack-PairMOT
ENV=/data/users/linxu/.conda/envs/py310_pairmot
RUNTIME=/data4/linxu/PairMOT_DiffusionTrack

export PATH="${ENV}/bin:/data/users/linxu/anaconda3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HSMOT_TRAIN_ROOT="${RUNTIME}/data/hsmot/train"
export HSMOT_VAL_ROOT="${RUNTIME}/data/hsmot/test"
export YOLO11_WEIGHTS="${RUNTIME}/pretrained_weights/mmot_official/yolo11L-8ch-3dstem.pt"

cd "${REPO}"
resume_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_args+=(--resume)
fi
exec python tools/train.py \
  -f exps/example/mot/yolo11l_diffusion_det_hsmot.py \
  -expn yolo11l_diffusion_det_hsmot_b4_d2_acc4_fp16_w8 \
  -d 2 -b 4 --accumulate 4 --fp16 \
  "${resume_args[@]}" \
  output_dir "${RUNTIME}/work_dirs" data_num_workers 8
