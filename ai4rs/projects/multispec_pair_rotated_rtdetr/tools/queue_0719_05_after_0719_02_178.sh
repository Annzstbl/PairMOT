#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
LAUNCHER=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/launch_0719_05_paper_base_rerun_178.sh
QUEUE_DIR=/data4/litianhao/PairMmot/workdir_178/0719_05_paper_base_rerun_r18_coco_full_1200x900_bf16_1xb8
LOCK=/data4/litianhao/PairMmot/workdir_178/.queue_0719_05.lock
PREDECESSOR=o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_liquid_pairconsensus_reliability_coco_full_1200x900_bf16_178.py
MEMORY_LIMIT_MIB=2048

mkdir -p "${QUEUE_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] another 0719_05 queue is already active"
    exit 3
fi

echo "[$(date '+%F %T')] waiting for 0719_02 and GPU 0"
ready_count=0
while true; do
    predecessor_active=0
    if pgrep -af "tools/train.py.*${PREDECESSOR}" >/dev/null; then
        predecessor_active=1
    fi

    used_mib=$(nvidia-smi --id=0 --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' ')
    gpu_ready=0
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
            && (( used_mib < MEMORY_LIMIT_MIB )); then
        gpu_ready=1
    fi

    if (( predecessor_active == 0 && gpu_ready == 1 )); then
        ((ready_count += 1))
    else
        ready_count=0
    fi
    if (( ready_count >= 3 )); then
        break
    fi
    echo "[$(date '+%F %T')] predecessor=${predecessor_active} gpu0_mem=${used_mib:-unavailable} MiB ready=${ready_count}/3"
    sleep 60
done

echo "[$(date '+%F %T')] conditions stable; starting 0719_05"
exec bash "${LAUNCHER}"
