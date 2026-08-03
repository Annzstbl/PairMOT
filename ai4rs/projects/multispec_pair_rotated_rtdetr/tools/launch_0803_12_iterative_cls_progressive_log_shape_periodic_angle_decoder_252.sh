#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/litianhao01/PairMmot_progressivegeom_0803_12_252/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_12_iterative_cls_progressive_log_shape_periodic_angle_decoder_252.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_252/0803_12_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_iterativeclsdnisolatede2e_pairsharedprogressivelogshape_periodicanglerefinement_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh

set +u
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data/users/litianhao01/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin

trap 'status=$?; echo "[$(date "+%F %T")] formal 0803_12 on 252 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

RESUME=${PAIRMOT_RESUME:-0}
if [[ "${RESUME}" == 1 ]]; then
    test -d "${WORK_DIR}"
    test -f "${WORK_DIR}/last_checkpoint"
else
    test ! -e "${WORK_DIR}"
    mkdir -p "${WORK_DIR}"
fi
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_test_gap1
export CUDA_VISIBLE_DEVICES=${PAIRMOT_CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

TRAIN_ARGS=(tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}")
if [[ "${RESUME}" == 1 ]]; then
    TRAIN_ARGS+=(--resume)
fi
echo "[$(date '+%F %T')] formal 0803_12 resume=${RESUME} GPUs=${CUDA_VISIBLE_DEVICES} commit $(git rev-parse --short HEAD)" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29861 \
    "${TRAIN_ARGS[@]}" \
    >> "${WORK_DIR}/launch.log" 2>&1
