#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/wangying01/lth/PairMOT_0810_01_stagedlr_resume_e68_99/ai4rs/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0810_01_iterative_cls_terminal_transport_product_tangent_staged_delayedlrclock_decoder_resume99.py
SOURCE_WORK_DIR=/data4/litianhao/PairMmot/workdir_197/0808_07_final_product_tangent_staged_delayedlrclock_72e_2xb4_fresh
WORK_DIR=/data4/litianhao/PairMmot/workdir_99/0810_01_final_product_tangent_staged_delayedlrclock_resume_e68_to_e72_2xb4
CHECKPOINT=${SOURCE_WORK_DIR}/epoch_68.pth
RESUME_LOG=${WORK_DIR}/resume_99_from_epoch68.log

set +u
source /data/users/wangying01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data/users/wangying01/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin

check_selected_gpus_idle() {
    local gpu
    IFS=',' read -r -a selected_gpus <<< "${PAIRMOT_CUDA_VISIBLE_DEVICES}"
    test "${#selected_gpus[@]}" -eq 2
    for gpu in "${selected_gpus[@]}"; do
        if nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
            echo "GPU ${gpu} is not idle" >&2
            return 1
        fi
    done
}

cd "${REPO}"
test -d "${SOURCE_WORK_DIR}"
test -s "${CHECKPOINT}"
: "${PAIRMOT_CUDA_VISIBLE_DEVICES:?set two currently free 99 GPU indices}"
check_selected_gpus_idle
sleep 5
check_selected_gpus_idle
test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
trap 'status=$?; echo "[$(date "+%F %T")] resume 0810_01 on 99 failed: status=${status} command=${BASH_COMMAND}" >> "${RESUME_LOG}"; exit "${status}"' ERR

test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/wangying01/lth/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data/users/wangying01/lth/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1
export CUDA_VISIBLE_DEVICES=${PAIRMOT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] resume 0808_07 epoch68 as 0810_01 on 99 GPUs=${CUDA_VISIBLE_DEVICES} commit $(git rev-parse --short HEAD)" >> "${RESUME_LOG}"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29903 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    --resume "${CHECKPOINT}" >> "${RESUME_LOG}" 2>&1
