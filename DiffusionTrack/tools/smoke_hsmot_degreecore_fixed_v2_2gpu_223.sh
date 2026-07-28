#!/usr/bin/env bash
# Ten real-image optimizer iterations plus distributed validation.
set -euo pipefail

REPO=/data/users/linxu/code/DiffusionTrack-PairMOT
ENV=/data/users/linxu/.conda/envs/py310_pairmot
RUNTIME=/data4/linxu/PairMOT_DiffusionTrack
EXP=smoke_yolo11l_diffusion_det_hsmot_degreecore_fixed_v2_2gpu
LOG=${RUNTIME}/logs/${EXP}_20260726.log

export PATH="${ENV}/bin:/data/users/linxu/anaconda3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_VISIBLE_DEVICES=0,3
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HSMOT_SMOKE_ROOT=${RUNTIME}/data/hsmot_overfit_data43_2_x20
export HSMOT_IMG_SUBDIR=npy2jpg
export HSMOT_IMG_FORMAT=3jpg
export YOLO11_WEIGHTS=${RUNTIME}/pretrained_weights/mmot_official/yolo11L-8ch-3dstem.pt

mkdir -p "${RUNTIME}/logs" "${RUNTIME}/work_dirs"
if [[ -e "${RUNTIME}/work_dirs/${EXP}" ]]; then
    echo "refusing to overwrite existing smoke workdir: ${RUNTIME}/work_dirs/${EXP}" >&2
    exit 1
fi

cd "${REPO}"
exec "${ENV}/bin/python" tools/train.py \
    -f exps/example/mot/yolo11l_diffusion_det_hsmot_degreecore_fixed_v2_smoke.py \
    -d 2 -b 2 --accumulate 1 --amp-dtype fp32 \
    --experiment-name "${EXP}" output_dir "${RUNTIME}/work_dirs" \
    >"${LOG}" 2>&1
