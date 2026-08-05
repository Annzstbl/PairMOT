#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/litianhao01/PairMOT_producttangent_extend_0806_04_252/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_88e_hsmot_0806_04_iterative_cls_terminal_transport_product_tangent_decoder_resume252.py
SOURCE_WORK_DIR=/data4/litianhao/PairMmot/workdir_252/0806_01_terminal_transport_product_tangent_resume252_e72_to_e80
WORK_DIR=/data4/litianhao/PairMmot/workdir_252/0806_04_terminal_transport_product_tangent_resume252_e80_to_e88
CHECKPOINT=${SOURCE_WORK_DIR}/epoch_80.pth
RESUME_LOG=${WORK_DIR}/resume_252_from_epoch80.log

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

cd "${REPO}"
test -d "${SOURCE_WORK_DIR}"
test -s "${CHECKPOINT}"
test ! -e "${WORK_DIR}"
check_fixed_gpus_idle
sleep 5
check_fixed_gpus_idle

mkdir -p "${WORK_DIR}"
trap 'status=$?; echo "[$(date "+%F %T")] resume 0806_04 on 252 failed: status=${status} command=${BASH_COMMAND}" >> "${RESUME_LOG}"; exit "${status}"' ERR
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_test_gap1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] resume mature 0806_04 from epoch80 to epoch88 on fixed 252 GPU0,1 commit $(git rev-parse --short HEAD)" >> "${RESUME_LOG}"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29904 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    --resume "${CHECKPOINT}" >> "${RESUME_LOG}" 2>&1
