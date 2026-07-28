#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 CONFIG.py EXP_NAME {bf16|fp32} [3jpg|npy]" >&2
    exit 2
fi

REPO=/data/users/linxu/code/DiffusionTrack-PairMOT
ENV=/data/users/linxu/.conda/envs/py310_pairmot
RUNTIME=/data4/linxu/PairMOT_DiffusionTrack
DATA=${RUNTIME}/data/hsmot_overfit_data43_2_x20
CONFIG=$1
EXP_NAME=$2
AMP_DTYPE=$3
IMG_FORMAT=${4:-3jpg}
LOG=${RUNTIME}/logs/${EXP_NAME}_20260724.log

if [[ "${AMP_DTYPE}" != "bf16" && "${AMP_DTYPE}" != "fp32" ]]; then
    echo "unsupported AMP dtype: ${AMP_DTYPE}" >&2
    exit 2
fi
if [[ "${IMG_FORMAT}" != "3jpg" && "${IMG_FORMAT}" != "npy" ]]; then
    echo "unsupported image format: ${IMG_FORMAT}" >&2
    exit 2
fi
if [[ "${IMG_FORMAT}" == "3jpg" ]]; then
    IMG_SUBDIR=npy2jpg
else
    IMG_SUBDIR=npy
fi

export PATH="${ENV}/bin:/data/users/linxu/anaconda3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HSMOT_OVERFIT_ROOT="${DATA}"
export HSMOT_IMG_SUBDIR="${IMG_SUBDIR}"
export HSMOT_IMG_FORMAT="${IMG_FORMAT}"
export YOLO11_WEIGHTS=${RUNTIME}/pretrained_weights/mmot_official/yolo11L-8ch-3dstem.pt

mkdir -p "${RUNTIME}/logs" "${RUNTIME}/work_dirs"
if [[ -e "${RUNTIME}/work_dirs/${EXP_NAME}" ]]; then
    echo "refusing to overwrite existing workdir: ${RUNTIME}/work_dirs/${EXP_NAME}" >&2
    exit 1
fi

cd "${REPO}"
exec "${ENV}/bin/python" tools/train.py \
    -f "exps/example/mot/${CONFIG}" \
    -d 1 -b 1 --accumulate 1 --amp-dtype "${AMP_DTYPE}" \
    --experiment-name "${EXP_NAME}" \
    output_dir "${RUNTIME}/work_dirs" \
    >"${LOG}" 2>&1
