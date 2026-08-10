#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/litianhao01/PairMOT_0810_07_ratio_onecycle200_252/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0810_07_iterative_cls_terminal_transport_product_tangent_ratio_onecycle200_decoder_252.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_252/0810_07_final_product_tangent_ratio_onecycle200_72e_2xb4_fresh

set +u
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data/users/litianhao01/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin

check_fixed_gpus_idle() {
    for gpu in 0 1; do
        if nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
            echo "GPU ${gpu} is not idle" >&2
            return 1
        fi
    done
}

trap 'status=$?; echo "[$(date "+%F %T")] formal 0810_07 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR
test ! -e "${WORK_DIR}"
check_fixed_gpus_idle
sleep 5
check_fixed_gpus_idle
mkdir -p "${WORK_DIR}"
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_test_gap1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh formal 0810_07 fixed GPUs=0,1 commit $(git rev-parse --short HEAD)" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29907 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
