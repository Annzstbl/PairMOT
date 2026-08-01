#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0731_01_sharedattention_antisymmetric_detail_decoder_178.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0731_01_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_antisymmetricdetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh
CHECKPOINT=${WORK_DIR}/epoch_8.pth

set +u
source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u

cd "${REPO}"
test -f "${CONFIG}"
test -d "${WORK_DIR}"
test -s "${CHECKPOINT}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset PYTORCH_CUDA_ALLOC_CONF
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] resume 0731_01 from epoch_8 commit $(git rev-parse --short HEAD)" \
    >> "${WORK_DIR}/launch.log"
exec python tools/train.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" --resume "${CHECKPOINT}" \
    >> "${WORK_DIR}/launch.log" 2>&1
