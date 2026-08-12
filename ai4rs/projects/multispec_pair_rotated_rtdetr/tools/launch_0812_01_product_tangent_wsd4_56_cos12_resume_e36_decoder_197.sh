#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/litianhao/PairMOT_0812_01_wsd_resume197/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0812_01_iterative_cls_terminal_transport_product_tangent_wsd4_56_cos12_decoder_resume197.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_197/0812_01_final_product_tangent_wsd4_56_cos12_72e_2xb4_resume_e36_to_e72_v2
RESUME=/data4/litianhao/PairMmot/workdir_252/0810_09_final_product_tangent_wsd4_56_cos12_72e_2xb4_fresh/epoch_36.pth

set +u
source /data/users/litianhao/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data/users/litianhao/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin

test -d "${REPO}"
test -f "${RESUME}"
test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
trap 'status=$?; echo "[$(date "+%F %T")] formal 0812_01 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/litianhao/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data/users/litianhao/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1
: "${PAIRMOT_CUDA_VISIBLE_DEVICES:=4,5}"
test "${PAIRMOT_CUDA_VISIBLE_DEVICES}" = 4,5
export CUDA_VISIBLE_DEVICES=${PAIRMOT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] exact resume 0812_01 GPUs=${CUDA_VISIBLE_DEVICES} commit $(git rev-parse --short HEAD) source=${RESUME}" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29919 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    --resume "${RESUME}" >> "${WORK_DIR}/launch.log" 2>&1
