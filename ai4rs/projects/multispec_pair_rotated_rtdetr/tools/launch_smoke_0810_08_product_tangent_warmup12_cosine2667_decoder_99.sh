#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/wangying01/lth/PairMOT_0810_08_warmup12_cosine2667_99/ai4rs/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_0810_08_product_tangent_warmup12_cosine2667_decoder_4iter_smoke_99.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_99/smoke_0810_08_product_tangent_warmup12_cosine2667_4iter

set +u
source /data/users/wangying01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data/users/wangying01/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin
trap 'status=$?; echo "[$(date "+%F %T")] smoke 0810_08 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

require_dir_with_retry() {
    local path=$1
    local attempt
    for attempt in $(seq 1 30); do
        if [[ -d "${path}" ]]; then
            return 0
        fi
        ls -ld "$(dirname "${path}")" >/dev/null 2>&1 || true
        sleep 1
    done
    echo "[$(date '+%F %T')] required directory remained unavailable after 30 attempts: ${path}" >> "${WORK_DIR}/launch.log"
    return 1
}

test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
require_dir_with_retry /data/users/wangying01/lth/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
require_dir_with_retry /data/users/wangying01/lth/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1
: "${PAIRMOT_CUDA_VISIBLE_DEVICES:?set two currently free 99 GPU indices}"
export CUDA_VISIBLE_DEVICES=${PAIRMOT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] 0810_08 smoke GPUs=${CUDA_VISIBLE_DEVICES} commit $(git rev-parse --short HEAD)" >> "${WORK_DIR}/launch.log"
"${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29909 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
if grep -Eiq 'Traceback|CUDA out of memory|loss: (nan|inf)|grad_norm: (nan|inf)|NCCL error|unused parameter.*error' "${WORK_DIR}/launch.log"; then exit 3; fi
test -s "${WORK_DIR}/iter_4.pth"
"${PYTHON_ROOT}/python" projects/multispec_pair_rotated_rtdetr/tools/check_iterative_cls_dn_isolated_checkpoint.py "${WORK_DIR}/iter_4.pth" >> "${WORK_DIR}/launch.log" 2>&1
"${PYTHON_ROOT}/python" projects/multispec_pair_rotated_rtdetr/tools/check_checkpoint_all_finite.py "${WORK_DIR}/iter_4.pth" >> "${WORK_DIR}/launch.log" 2>&1
