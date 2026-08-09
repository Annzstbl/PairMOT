#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data1/users/litianhao01/PairMOT_0810_02_ema_lag_eval_178/ai4rs/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0809_01_iterative_cls_terminal_transport_product_tangent_decoderhead_delayedlrclock_decoder_178.py
SOURCE_ROOT=/data4/litianhao/PairMmot/workdir_252/0809_03_product_tangent_epoch72_ema_lag_correction_eval
ROOT=/data4/litianhao/PairMmot/workdir_178/0810_02_product_tangent_epoch72_ema_lag_correction_eval
VERIFY=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/verify_decoder_epoch72_goal.py

set +u
source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data1/users/litianhao01/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin

: "${PAIRMOT_CUDA_VISIBLE_DEVICES:?set one currently free 178 GPU index}"
[[ "${PAIRMOT_CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]
check_gpu_idle() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -i "${PAIRMOT_CUDA_VISIBLE_DEVICES}" | tr -d ' ')
    if (( used >= 500 )); then
        echo "GPU ${PAIRMOT_CUDA_VISIBLE_DEVICES} is occupied: ${used} MiB" >&2
        return 1
    fi
}
check_gpu_idle
sleep 5
check_gpu_idle

test -d "${REPO}/../.git"
test -f "${REPO}/${CONFIG}"
test -f "${VERIFY}"
test ! -e "${ROOT}"
test -s "${SOURCE_ROOT}/checkpoints/online_fraction_025.pth"
test -s "${SOURCE_ROOT}/checkpoints/online_fraction_050.pth"
mkdir -p "${ROOT}"
trap 'status=$?; echo "[$(date "+%F %T")] 0810_02 failed: status=${status} command=${BASH_COMMAND}" >> "${ROOT}/runner.log"; exit "${status}"' ERR

export CUDA_VISIBLE_DEVICES=${PAIRMOT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO}"
echo "[$(date '+%F %T')] 0810_02 serial EMA-lag evaluation GPU=${CUDA_VISIBLE_DEVICES} commit $(git rev-parse --short HEAD)" >> "${ROOT}/runner.log"

run_one() {
    local fraction=$1
    local checkpoint=${SOURCE_ROOT}/checkpoints/online_fraction_${fraction}.pth
    local output=${ROOT}/online_fraction_${fraction}
    local metrics=${output}/val_track_eval/val_track_0001/metrics.json
    test ! -e "${output}"
    mkdir -p "${output}"
    ln -s "${checkpoint}" "${output}/epoch_72.pth"
    echo "[$(date '+%F %T')] fraction=${fraction} test start" >> "${ROOT}/runner.log"
    "${PYTHON_ROOT}/python" tools/test.py "${CONFIG}" "${checkpoint}" \
        --work-dir "${output}" \
        --cfg-options \
        "test_evaluator.metrics.track_eval_out_dir=${output}/val_track_eval" \
        "test_evaluator.metrics.val_det_out_dir=${output}/val_det" \
        test_evaluator.metrics.async_track_eval=True \
        > "${output}/test.log" 2>&1
    for _ in $(seq 1 120); do
        [[ -s "${metrics}" ]] && break
        sleep 15
    done
    test -s "${metrics}"
    set +e
    "${PYTHON_ROOT}/python" "${VERIFY}" "${output}" --epoch 72 \
        --payload-step 1 --json-out "${output}/strict_verification.json" \
        >> "${ROOT}/runner.log" 2>&1
    local verify_status=$?
    set -e
    if (( verify_status != 0 && verify_status != 2 )); then
        return "${verify_status}"
    fi
    echo "[$(date '+%F %T')] fraction=${fraction} verifier=${verify_status}" >> "${ROOT}/runner.log"
}

run_one 025
run_one 050
echo "[$(date '+%F %T')] 0810_02 complete" >> "${ROOT}/runner.log"
