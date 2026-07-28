#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
SOURCE_HSMOT_ROOT=/data1/users/litianhao01/data/hsmot
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0725_02_paper_liquid_pairdn_paircoherent_le180_dse_cpdse_centered_r18_coco_full_1200x900_bf16_1xb4acc2_protocolfix_fresh
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb4acc2_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_dse_cpdse_centered_coco_full_1200x900_bf16_178.py
CACHE_SCRIPT=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/prepare_hsmot_shm_cache_178.sh
LOG=${WORK_DIR}/launch.log

mkdir -p "${WORK_DIR}"
if find "${WORK_DIR}" -mindepth 1 -maxdepth 1 ! -name launch.log | grep -q .; then
    echo "Refusing a fresh launch into non-empty ${WORK_DIR}" >&2
    exit 2
fi

set +u
source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1

export PAIRMOT_HSMOT_ROOT="${SOURCE_HSMOT_ROOT}"
if cache_root=$(bash "${CACHE_SCRIPT}" 2>> "${LOG}"); then
    export PAIRMOT_HSMOT_ROOT="${cache_root}"
    echo "[$(date '+%F %T')] using validated tmpfs image cache ${cache_root}" >> "${LOG}"
else
    echo "[$(date '+%F %T')] tmpfs unavailable; using local NVMe ${SOURCE_HSMOT_ROOT}" >> "${LOG}"
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0725_02 PairDN + DSE + centered pair-global CP-DSE, bs=4, accum=2, warmup=4000, EMA interval=2 gamma=4000" >> "${LOG}"
python tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" >> "${LOG}" 2>&1
