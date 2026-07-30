#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0730_04_decoder_boxonly_gradisolated_reg1p0_178.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0730_04_decoder_boxonly_gradisolated_reg1p0_pairdn_paircoherent_1xb8_fresh
SOURCE_PID=1784051

while kill -0 "${SOURCE_PID}" 2>/dev/null; do
    sleep 30
done
while ps -eo args= | grep -q '[t]ools/train.py'; do
    sleep 10
done

test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0730_04 commit e410a58 after completed 0729_05" \
    >> "${WORK_DIR}/launch.log"
exec /data1/users/litianhao01/anaconda3/envs/py310/bin/python \
    tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
