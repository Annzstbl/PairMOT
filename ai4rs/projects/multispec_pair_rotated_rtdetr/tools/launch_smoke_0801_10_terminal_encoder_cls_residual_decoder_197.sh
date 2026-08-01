#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data/users/litianhao/PairMOT_sync_3cb888d/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_0801_10_terminal_encoder_cls_residual_decoder_4iter_smoke_197.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_197/smoke_0801_10_terminal_encoder_cls_residual_decoder_4iter
PYTHON_ROOT=/data/users/litianhao/anaconda3/envs/py310/bin

test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/litianhao/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] smoke 0801_10 commit $(git rev-parse --short HEAD)" \
    >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29984 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
