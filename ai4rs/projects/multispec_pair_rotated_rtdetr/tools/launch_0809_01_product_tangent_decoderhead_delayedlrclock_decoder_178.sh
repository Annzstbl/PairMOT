#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data1/users/litianhao01/PairMOT_0809_01_decoderhead_delayedlr_178/ai4rs/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0809_01_iterative_cls_terminal_transport_product_tangent_decoderhead_delayedlrclock_decoder_178.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0809_01_final_product_tangent_decoderhead_delayedlrclock_72e_1xb8_fresh

set +u
source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data1/users/litianhao01/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin
trap 'status=$?; echo "[$(date "+%F %T")] formal 0809_01 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1
: "${PAIRMOT_CUDA_VISIBLE_DEVICES:?set one currently free 178 GPU index}"
export CUDA_VISIBLE_DEVICES=${PAIRMOT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh formal 0809_01 GPU=${CUDA_VISIBLE_DEVICES} commit $(git rev-parse --short HEAD)" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/python" tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" >> "${WORK_DIR}/launch.log" 2>&1
