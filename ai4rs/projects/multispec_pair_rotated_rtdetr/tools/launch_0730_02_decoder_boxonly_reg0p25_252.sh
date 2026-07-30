#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data/users/litianhao01/PairMmot/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0730_02_decoder_boxonly_gradisolated_reg0p25_252.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_252/0730_02_decoder_boxonly_gradisolated_reg0p25_pairdn_paircoherent_2xb4_fresh
PYTHON_ROOT=/data/users/litianhao01/anaconda3/envs/py310/bin

test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0730_02 commit e410a58" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29538 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
