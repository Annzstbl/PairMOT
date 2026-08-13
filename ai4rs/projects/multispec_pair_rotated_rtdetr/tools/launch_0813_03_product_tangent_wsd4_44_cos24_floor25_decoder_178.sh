#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data1/users/litianhao01/PairMOT_0812_04_wsd44_cos24_178/ai4rs/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0813_03_iterative_cls_terminal_transport_product_tangent_wsd4_44_cos24_floor25_decoder_178.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0813_03_final_product_tangent_wsd4_44_cos24_floor25_72e_1xb8_fresh

set +u
source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
trap 'status=$?; echo "[$(date "+%F %T")] formal 0813_03 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR
test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG
echo "[$(date '+%F %T')] fresh formal 0813_03 GPU=${CUDA_VISIBLE_DEVICES}" >> "${WORK_DIR}/launch.log"
exec "${CONDA_PREFIX}/bin/python" tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
