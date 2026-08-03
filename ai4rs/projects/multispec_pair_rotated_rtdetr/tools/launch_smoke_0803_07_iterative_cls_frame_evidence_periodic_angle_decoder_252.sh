#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/litianhao01/PairMmot_framecls_periodic_0803_07/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_0803_07_iterative_cls_frame_evidence_periodic_angle_decoder_4iter_smoke_252.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_252/smoke_0803_07_iterative_cls_frame_evidence_periodic_angle_4iter

set +u
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data/users/litianhao01/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin

trap 'status=$?; echo "[$(date "+%F %T")] smoke 0803_07 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_train_gap1
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0803_07 real-data DDP smoke commit $(git rev-parse --short HEAD)" >> "${WORK_DIR}/launch.log"
"${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29848 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
if grep -Eiq \
        'Traceback|CUDA out of memory|loss: (nan|inf)|grad_norm: (nan|inf)|NCCL error|unused parameter.*error' \
        "${WORK_DIR}/launch.log"; then
    exit 3
fi
test -f "${WORK_DIR}/iter_4.pth"
"${PYTHON_ROOT}/python" \
    projects/multispec_pair_rotated_rtdetr/tools/check_iterative_cls_dn_isolated_checkpoint.py \
    "${WORK_DIR}/iter_4.pth" >> "${WORK_DIR}/launch.log" 2>&1
