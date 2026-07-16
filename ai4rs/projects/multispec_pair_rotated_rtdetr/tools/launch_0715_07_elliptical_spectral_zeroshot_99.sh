#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/wangying01/lth/PairMOT/ai4rs
WORK_DIR=/data4/litianhao/PairMmot/workdir_99/0715_07_full_baseline_elliptical_spectral_zeroshot
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_elliptical_spectral_zeroshot_99.py
CHECKPOINT=/data4/litianhao/PairMmot/workdir_252/0714_01_0704_resume_coco365_full_unique_allgt/epoch_72.pth
LOG=${WORK_DIR}/test.log

mkdir -p "${WORK_DIR}"
source /data/users/wangying01/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd "${REPO}"

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] zero-shot test ${CONFIG} on GPU ${CUDA_VISIBLE_DEVICES}" >> "${LOG}"
python tools/test.py "${CONFIG}" "${CHECKPOINT}" \
    --work-dir "${WORK_DIR}" >> "${LOG}" 2>&1
