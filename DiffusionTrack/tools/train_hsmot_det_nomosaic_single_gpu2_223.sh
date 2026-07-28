#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/linxu/code/DiffusionTrack-PairMOT
ENV=/data/users/linxu/.conda/envs/py310_pairmot
RUNTIME=/data4/linxu/PairMOT_DiffusionTrack
EXP_NAME=yolo11l_diffusion_det_hsmot_nomosaic_fixed896x1184_b2_d1_acc2_bf16_plainriou_v4
LOG=${RUNTIME}/logs/stage1_det_nomosaic_fixed896x1184_b2_d1_acc2_bf16_plainriou_v4_20260723.log

export PATH="${ENV}/bin:/data/users/linxu/anaconda3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HSMOT_TRAIN_ROOT=/data/users/qinhaolin01/data/hsmot/train
export HSMOT_VAL_ROOT=/data/users/qinhaolin01/data/hsmot/test
export HSMOT_IMG_SUBDIR=npy2jpg
export HSMOT_IMG_FORMAT=3jpg
export YOLO11_WEIGHTS=${RUNTIME}/pretrained_weights/mmot_official/yolo11L-8ch-3dstem.pt

mkdir -p "${RUNTIME}/logs"
if [[ -e "${RUNTIME}/work_dirs/${EXP_NAME}" ]]; then
    echo "refusing to overwrite existing workdir: ${RUNTIME}/work_dirs/${EXP_NAME}" >&2
    exit 1
fi

cd "${REPO}"
exec "${ENV}/bin/python" tools/train.py \
    -f exps/example/mot/yolo11l_diffusion_det_hsmot_nomosaic_fixed896x1184.py \
    -d 1 -b 2 --accumulate 2 --amp-dtype bf16 \
    --experiment-name "${EXP_NAME}" \
    output_dir "${RUNTIME}/work_dirs" \
    >"${LOG}" 2>&1
