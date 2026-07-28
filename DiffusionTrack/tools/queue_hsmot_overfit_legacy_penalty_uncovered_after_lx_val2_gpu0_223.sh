#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/linxu/code/DiffusionTrack-PairMOT
LX_RUNTIME=/data4/linxu/PairMOT_DiffusionTrack/lx_baseline_isolated
RUNTIME=/data4/linxu/PairMOT_DiffusionTrack
PREV_PID=79485
PREV_LOG=${LX_RUNTIME}/logs/lx_baseline_diffusiontrack_single_data43_2_x20_gpu0_val2_v1.log
QUEUE_LOG=${RUNTIME}/logs/queue_stage1_overfit_legacycenter_uncovered_after_lx_val2_gpu0_v1.log
LAUNCHER=${REPO}/tools/train_hsmot_overfit_legacy_penalty_uncovered_gpu0_223.sh
GPU0_UUID=GPU-29ca0f83-c9e4-82ec-8331-2c7fd99445dd

mkdir -p "${RUNTIME}/logs"
exec >>"${QUEUE_LOG}" 2>&1
trap 'rc=$?; echo "$(date "+%F %T") FAILED rc=${rc} command=${BASH_COMMAND}"' ERR

echo "$(date "+%F %T") QUEUED waiting_for_pid=${PREV_PID}"
while kill -0 "${PREV_PID}" 2>/dev/null; do
    sleep 20
done

if ! grep -q "Training of experiment is done" "${PREV_LOG}"; then
    echo "$(date "+%F %T") predecessor did not complete successfully"
    exit 1
fi

idle_checks=0
while (( idle_checks < 3 )); do
    used_mb="$(nvidia-smi --query-compute-apps=used_memory,gpu_uuid \
        --format=csv,noheader,nounits \
        | awk -F', *' -v uuid="${GPU0_UUID}" \
            '$2 == uuid {sum += $1} END {print sum + 0}')"
    if (( used_mb < 1024 )); then
        idle_checks=$((idle_checks + 1))
    else
        idle_checks=0
    fi
    echo "$(date "+%F %T") gpu0_used_mb=${used_mb} idle_checks=${idle_checks}/3"
    sleep 10
done

echo "$(date "+%F %T") STARTING ${LAUNCHER}"
exec bash "${LAUNCHER}"
