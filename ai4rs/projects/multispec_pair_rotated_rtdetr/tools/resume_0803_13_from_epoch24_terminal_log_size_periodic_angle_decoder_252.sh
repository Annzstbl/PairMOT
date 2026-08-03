#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/litianhao01/PairMmot_terminal_0803_13_resume252/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_13_iterative_cls_terminal_log_size_periodic_angle_decoder_resume252.py
SOURCE_WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0803_13_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_iterativeclsdnisolatede2e_pairsharedterminallogsizetangent_periodicanglerefinement_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh
WORK_DIR=/data4/litianhao/PairMmot/workdir_252/0803_13_terminal_log_size_periodic_angle_resume252_from_epoch24
CHECKPOINT=${SOURCE_WORK_DIR}/epoch_24.pth
RESUME_LOG=${WORK_DIR}/resume_252_from_epoch24.log

set +u
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data/users/litianhao01/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin

cd "${REPO}"
test -d "${SOURCE_WORK_DIR}"
test -s "${CHECKPOINT}"
test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
trap 'status=$?; echo "[$(date "+%F %T")] resume 0803_13 on 252 failed: status=${status} command=${BASH_COMMAND}" >> "${RESUME_LOG}"; exit "${status}"' ERR
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_test_gap1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] resume mature 0803_13 from epoch24 on fixed 252 GPU0,1 commit $(git rev-parse --short HEAD)" >> "${RESUME_LOG}"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29871 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    --resume "${CHECKPOINT}" >> "${RESUME_LOG}" 2>&1
