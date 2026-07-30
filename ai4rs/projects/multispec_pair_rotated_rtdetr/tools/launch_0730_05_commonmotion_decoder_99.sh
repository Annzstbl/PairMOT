#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data/users/wangying01/lth/PairMOT/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0730_05_commonmotion_decoder_pairdn_paircoherent_99.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_99/0730_05_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_commonmotion_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_orderedpairs_fresh
PYTHON_ROOT=/data/users/wangying01/anaconda3/envs/py310/bin

test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/wangying01/lth/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data/users/wangying01/lth/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh formal 0730_05" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29546 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
