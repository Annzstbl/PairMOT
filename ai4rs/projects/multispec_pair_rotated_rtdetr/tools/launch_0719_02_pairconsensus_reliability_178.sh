#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0719_02_paper_liquid_pairconsensus_reliability_r18_coco_full_1200x900_bf16_1xb8
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_liquid_pairconsensus_reliability_coco_full_1200x900_bf16_178.py
LOG=${WORK_DIR}/launch.log

mkdir -p "${WORK_DIR}"
if find "${WORK_DIR}" -mindepth 1 -maxdepth 1 ! -name launch.log | grep -q .; then
    echo "Refusing a fresh launch into non-empty ${WORK_DIR}" >&2
    exit 2
fi

source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd "${REPO}"

test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh reliability-weighted pair-consensus Liquid on GPU ${CUDA_VISIBLE_DEVICES}, bs=8, workers=8" >> "${LOG}"
python tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" >> "${LOG}" 2>&1
