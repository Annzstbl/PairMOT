#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data/users/litianhao01/PairMmot/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_05_sharedattention_enveloped_detail_decoder_252.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_252/0731_05_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_envelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh
CHECKPOINT=${WORK_DIR}/epoch_16.pth

trap 'status=$?; echo "[$(date "+%F %T")] resume 0731_05 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

set +u
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u

cd "${REPO}"
test -f "${CONFIG}"
test -d "${WORK_DIR}"
test -s "${CHECKPOINT}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_test_gap1

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] resume 0731_05 from epoch_16 commit $(git rev-parse --short HEAD)" \
    >> "${WORK_DIR}/launch.log"
exec torchrun --nproc_per_node=2 --master_port=29566 \
    tools/train.py "${CONFIG}" --launcher pytorch \
    --work-dir "${WORK_DIR}" --resume "${CHECKPOINT}" \
    >> "${WORK_DIR}/launch.log" 2>&1
