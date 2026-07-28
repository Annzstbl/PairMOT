#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
SOURCE_HSMOT_ROOT=/data1/users/litianhao01/data/hsmot
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/smoke_0729_05_liquid_independent_diffproduct_pairdn_easyhardpositive_4iter
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_paper_liquid_independent_diffproduct_pairdn_easyhardpositive_4iter_smoke_178.py
CACHE_SCRIPT=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/prepare_hsmot_shm_cache_178.sh
LOG=${WORK_DIR}/launch.log

mkdir -p "${WORK_DIR}"
if find "${WORK_DIR}" -mindepth 1 -maxdepth 1 ! -name launch.log | grep -q .; then
    echo "Refusing a fresh smoke launch into non-empty ${WORK_DIR}" >&2
    exit 2
fi

set +u
source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
cd "${REPO}"

export PAIRMOT_HSMOT_ROOT="${SOURCE_HSMOT_ROOT}"
if cache_root=$(bash "${CACHE_SCRIPT}" 2>> "${LOG}"); then
    export PAIRMOT_HSMOT_ROOT="${cache_root}"
fi
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0729_05 4-iter smoke" >> "${LOG}"
python tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" >> "${LOG}" 2>&1

if grep -Eiq \
        'Traceback|CUDA out of memory|loss: (nan|inf)|grad_norm: (nan|inf)' \
        "${LOG}"; then
    echo "Smoke log contains a fatal or non-finite signal" >&2
    exit 3
fi
test -f "${WORK_DIR}/iter_4.pth"
echo "PAIRDNEASYPOS_SMOKE_OK" >> "${LOG}"
