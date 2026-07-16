#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/litianhao01/PairMmot/ai4rs
WORK_DIR=/data4/litianhao/PairMmot/workdir_252/0715_06_liquid8_pairbandcontext_paironly_coco365_full_bf16
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_pairbandcontext_paironly_coco365_full_bf16_252.py
LOG=${WORK_DIR}/launch.log

mkdir -p "${WORK_DIR}"
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd "${REPO}"

export CUDA_VISIBLE_DEVICES=0,1
export PORT=29878
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh launch ${CONFIG} on GPUs ${CUDA_VISIBLE_DEVICES}, port ${PORT}" >> "${LOG}"
bash tools/dist_train.sh "${CONFIG}" 2 --work-dir "${WORK_DIR}" >> "${LOG}" 2>&1
