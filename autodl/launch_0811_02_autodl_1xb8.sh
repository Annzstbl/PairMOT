#!/usr/bin/env bash
set -Eeuo pipefail

set +u
source /root/PairMOT/autodl_runtime.env
set -u

REPO=${PAIRMOT_AUTODL_REPO:-/root/PairMOT_0811_02_autodl/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/autodl_0811_02_product_tangent_warmup4_cosine2667_72e_1xb8.py
WORK_DIR=/root/autodl-tmp/work_dirs/0811_02_final_product_tangent_warmup4_cosine2667_72e_1xb8_autodl_fresh_v2
PYTHON_BIN=/root/miniconda3/bin/python

trap 'status=$?; echo "[$(date "+%F %T")] formal 0811_02 AutoDL failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

test ! -e "${WORK_DIR}"
test -f /root/autodl-fs/PairMOT_assets/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /root/autodl-tmp/data/hsmot/train/mot
test -d /root/autodl-tmp/data/hsmot/train/npy2jpg
test -d /root/autodl-tmp/data/hsmot/test/mot
test -d /root/autodl-tmp/data/hsmot/test/npy2jpg
test -d /root/autodl-tmp/PairMOT_assets/gmc_cache/hsmot_train_gap1
test -d /root/autodl-tmp/PairMOT_assets/gmc_cache/hsmot_test_gap1

mkdir -p "${WORK_DIR}"
echo "$$" > "${WORK_DIR}/launcher.pid"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHON="${PYTHON_BIN}"
export PATH=/root/miniconda3/bin:${PATH}
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0811_02 AutoDL GPU0 physical batch 8 commit $(git rev-parse --short HEAD)" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_BIN}" tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" >> "${WORK_DIR}/launch.log" 2>&1
