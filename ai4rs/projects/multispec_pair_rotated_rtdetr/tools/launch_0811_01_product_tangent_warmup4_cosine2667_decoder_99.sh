#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/wangying01/lth/PairMOT_0811_01_warmup4_cosine2667_99/ai4rs/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0811_01_iterative_cls_terminal_transport_product_tangent_warmup4_cosine2667_decoder_99.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_99/0811_01_final_product_tangent_warmup4_cosine2667_72e_2xb4_fresh

set +u
source /data/users/wangying01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data/users/wangying01/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin
trap 'status=$?; echo "[$(date "+%F %T")] formal 0811_01 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

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

echo "[$(date '+%F %T')] fresh formal 0811_01 GPUs=${CUDA_VISIBLE_DEVICES} commit $(git rev-parse --short HEAD)" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29916 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
