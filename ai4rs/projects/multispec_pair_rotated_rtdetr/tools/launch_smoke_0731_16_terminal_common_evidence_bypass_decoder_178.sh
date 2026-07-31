#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_0731_16_terminal_common_evidence_bypass_decoder_4iter_smoke_178.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/smoke_0731_16_terminal_common_evidence_bypass_decoder_4iter
PYTHON_ROOT=/data1/users/litianhao01/anaconda3/envs/py310/bin

test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0731_16 real-data smoke" \
    >> "${WORK_DIR}/launch.log"
"${PYTHON_ROOT}/python" tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1

if grep -Eiq \
        'Traceback|CUDA out of memory|loss: (nan|inf)|grad_norm: (nan|inf)' \
        "${WORK_DIR}/launch.log"; then
    echo "Smoke log contains a fatal or non-finite signal" >&2
    exit 3
fi
test -f "${WORK_DIR}/iter_4.pth"
"${PYTHON_ROOT}/python" \
    projects/multispec_pair_rotated_rtdetr/tools/check_terminal_common_evidence_bypass_checkpoint.py \
    "${WORK_DIR}/iter_4.pth" >> "${WORK_DIR}/launch.log" 2>&1
