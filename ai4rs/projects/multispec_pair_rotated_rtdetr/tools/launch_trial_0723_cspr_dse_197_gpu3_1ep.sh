#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/litianhao/PairMOT/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/tmp_trial_0723_cspr_dse_197_1xb8_1ep_shm.py
CACHE_SCRIPT=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/prepare_hsmot_shm_cache_197.sh
WORK_DIR=/data4/litianhao/PairMmot/workdir_197/trial_0723_cspr_dse_pairdn_gpu3_1xb8_1ep_shm

if [[ -e "${WORK_DIR}" ]]; then
    echo "Refusing to overwrite existing trial: ${WORK_DIR}" >&2
    exit 20
fi
mkdir -p "${WORK_DIR}"

set +u
source /data/users/litianhao/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u

export PAIRMOT_HSMOT_ROOT
PAIRMOT_HSMOT_ROOT=$(bash "${CACHE_SCRIPT}")
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd "${REPO}"
echo "[$(date '+%F %T')] CSPR-DSE one-epoch trial started" \
    | tee -a "${WORK_DIR}/launcher.log"
python tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" \
    2>&1 | tee -a "${WORK_DIR}/launch.log"
echo "[$(date '+%F %T')] CSPR-DSE one-epoch trial completed" \
    | tee -a "${WORK_DIR}/launcher.log"
