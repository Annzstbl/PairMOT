#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data/users/litianhao/PairMOT_sync_3cb888d/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_09_iterative_cls_dn_isolated_e2e_decoder_resume197.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0801_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_iterativeclsdnisolatede2e_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh
CHECKPOINT=${WORK_DIR}/epoch_56.pth

set +u
source /data/users/litianhao/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data/users/litianhao/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin

test -f "${CHECKPOINT}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] resume 0801_09 from epoch_56 on physical 2x4 commit $(git rev-parse --short HEAD)" >> "${WORK_DIR}/resume_197.log"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29991 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    --resume "${CHECKPOINT}" >> "${WORK_DIR}/resume_197.log" 2>&1
