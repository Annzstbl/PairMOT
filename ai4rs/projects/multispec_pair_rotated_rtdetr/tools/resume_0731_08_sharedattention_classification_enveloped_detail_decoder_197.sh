#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data/users/litianhao/PairMOT_sync_3cb888d/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_08_sharedattention_classification_enveloped_detail_decoder_197.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_197/0731_08_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_classificationenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh
CHECKPOINT="${WORK_DIR}/epoch_4.pth"
PYTHON_ROOT=/data/users/litianhao/anaconda3/envs/py310/bin

test -f "${CHECKPOINT}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] resume 0731_08 from epoch_4; HOTA-primary gate" \
    >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29967 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    --resume "${CHECKPOINT}" >> "${WORK_DIR}/launch.log" 2>&1
