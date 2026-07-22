#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/wangying01/lth/PairMOT/ai4rs
WORK_DIR=/data4/litianhao/PairMmot/workdir_99/0718_05_paper_liquid_adaptiveanchor_pcdp_r18_coco_full_1200x900_bf16_orderedpairs_fresh
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_adaptiveanchor_pcdp_coco_full_1200x900_bf16_99.py
LOG=${WORK_DIR}/launch.log

mkdir -p "${WORK_DIR}"
if find "${WORK_DIR}" -mindepth 1 -maxdepth 1 ! -name launch.log | grep -q .; then
    echo "Refusing a fresh launch into non-empty ${WORK_DIR}" >&2
    exit 2
fi

source /data/users/wangying01/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd "${REPO}"

export CUDA_VISIBLE_DEVICES=0,1
export PORT=29895
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh PCDP-Liquid on GPUs ${CUDA_VISIBLE_DEVICES}, port ${PORT}" >> "${LOG}"
bash tools/dist_train.sh "${CONFIG}" 2 \
    --work-dir "${WORK_DIR}" >> "${LOG}" 2>&1
