#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data/users/wangying01/lth/PairMOT_0812_03_cosine_floor50_99/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0813_02_iterative_cls_terminal_transport_product_tangent_warmup4_cosine_floor65_decoder_99.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_99/0813_02_final_product_tangent_warmup4_cosine_floor65_72e_2xb4_fresh_v2

set +u
source /data/users/wangying01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
trap 'status=$?; echo "[$(date "+%F %T")] formal 0813_02 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR
test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=0,2
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG
echo "[$(date '+%F %T')] fresh formal 0813_02 GPUs=${CUDA_VISIBLE_DEVICES}" >> "${WORK_DIR}/launch.log"
exec "${CONDA_PREFIX}/bin/torchrun" --nproc_per_node=2 --master_port=29932 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
