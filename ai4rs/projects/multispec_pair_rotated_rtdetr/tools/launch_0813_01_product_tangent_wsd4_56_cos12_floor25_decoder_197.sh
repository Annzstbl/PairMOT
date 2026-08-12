#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/litianhao/PairMOT_0813_01_wsd_floor25_197/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0813_01_iterative_cls_terminal_transport_product_tangent_wsd4_56_cos12_floor25_decoder_197.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_197/0813_01_final_product_tangent_wsd4_56_cos12_floor25_72e_2xb4_fresh

set +u
source /data/users/litianhao/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
PYTHON_ROOT=${CONDA_PREFIX}/bin

test -d "${REPO}"
test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
trap 'status=$?; echo "[$(date "+%F %T")] formal 0813_01 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

cd "${REPO}"
: "${PAIRMOT_CUDA_VISIBLE_DEVICES:=4,5}"
export CUDA_VISIBLE_DEVICES=${PAIRMOT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh formal 0813_01 GPUs=${CUDA_VISIBLE_DEVICES} commit $(git rev-parse --short HEAD)" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29931 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
