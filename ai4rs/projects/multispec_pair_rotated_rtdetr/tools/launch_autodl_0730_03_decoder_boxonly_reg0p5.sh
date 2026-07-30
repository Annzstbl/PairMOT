#!/usr/bin/env bash
set -Eeuo pipefail

source /root/PairMOT/autodl_runtime.env

REPO=/root/PairMOT/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/autodl_0730_03_decoder_boxonly_gradisolated_reg0p5_pairdn_paircoherent_1xb8.py
WORK_DIR=/root/autodl-tmp/work_dirs/0730_03_decoder_boxonly_gradisolated_reg0p5_pairdn_paircoherent_1xb8_fresh
PYTHON_BIN=/root/miniconda3/bin/python3.12

test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=0
export PYTHON="${PYTHON_BIN}"
export PATH=/root/miniconda3/bin:${PATH}
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0730_03 commit e410a58" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_BIN}" tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
