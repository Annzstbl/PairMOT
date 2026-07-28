#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/linxu/code/DiffusionTrack-lx-baseline-isolated
ENV=/data/users/linxu/.conda/envs/py310_pairmot
RUNTIME=/data4/linxu/PairMOT_DiffusionTrack/lx_baseline_isolated
DATA=${RUNTIME}/data
EXP_NAME=lx_baseline_diffusiontrack_single_data43_2_x20_gpu0_val2_thresh001_diag_v3
LOG=${RUNTIME}/logs/${EXP_NAME}.log

export PATH="${ENV}/bin:/data/users/linxu/anaconda3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export YOLOX_DATADIR="${DATA}"
export PYTHONPATH="${REPO}/detectron2:${PYTHONPATH:-}"

mkdir -p "${RUNTIME}/logs" "${RUNTIME}/outputs"
if [[ -e "${RUNTIME}/outputs/${EXP_NAME}" ]]; then
    echo "refusing to overwrite existing output: ${RUNTIME}/outputs/${EXP_NAME}" >&2
    exit 1
fi

cd "${REPO}"
exec "${ENV}/bin/python" tools/train.py \
    -f exps/example/mot/yolo11l_diffusion_det_mmot_single_image_overfit_val2_thresh001.py \
    -d 1 -b 1 \
    --experiment-name "${EXP_NAME}" \
    output_dir "${RUNTIME}/outputs" \
    >"${LOG}" 2>&1
