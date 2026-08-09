#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data1/users/litianhao01/PairMOT_0810_03_decoder_frozen_ema_lag_eval_178/ai4rs/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0809_01_iterative_cls_terminal_transport_product_tangent_decoderhead_delayedlrclock_decoder_178.py
ROOT=/data4/litianhao/PairMmot/workdir_178/0810_03_product_tangent_epoch72_decoder_frozen_ema_lag_eval
CHECKPOINT=${ROOT}/checkpoints/online_fraction_025_exclude_decoder.pth
OUTPUT=${ROOT}/online_fraction_025_exclude_decoder
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
test -s "${CHECKPOINT}"
test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT}"
trap 'status=$?; echo "[$(date "+%F %T")] 0810_03 failed: status=${status} command=${BASH_COMMAND}" >> "${ROOT}/runner.log"; exit "${status}"' ERR

export CUDA_VISIBLE_DEVICES=${PAIRMOT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO}"
ln -s "${CHECKPOINT}" "${OUTPUT}/epoch_72.pth"
echo "[$(date '+%F %T')] 0810_03 decoder-frozen EMA-lag test GPU=${CUDA_VISIBLE_DEVICES} commit $(git rev-parse --short HEAD)" >> "${ROOT}/runner.log"
"${PYTHON_ROOT}/python" tools/test.py "${CONFIG}" "${CHECKPOINT}" \
    --work-dir "${OUTPUT}" \
    --cfg-options \
    "test_evaluator.metrics.track_eval_out_dir=${OUTPUT}/val_track_eval" \
    "test_evaluator.metrics.val_det_out_dir=${OUTPUT}/val_det" \
    test_evaluator.metrics.async_track_eval=True \
    > "${OUTPUT}/test.log" 2>&1

metrics=${OUTPUT}/val_track_eval/val_track_0001/metrics.json
for _ in $(seq 1 120); do
    [[ -s "${metrics}" ]] && break
    sleep 15
done
test -s "${metrics}"
if "${PYTHON_ROOT}/python" "${VERIFY}" "${OUTPUT}" --epoch 72 \
        --payload-step 1 --json-out "${OUTPUT}/strict_verification.json" \
        >> "${ROOT}/runner.log" 2>&1; then
    verify_status=0
else
    verify_status=$?
fi
if (( verify_status != 0 && verify_status != 2 )); then
    exit "${verify_status}"
fi
echo "[$(date '+%F %T')] 0810_03 verifier=${verify_status} complete" >> "${ROOT}/runner.log"
