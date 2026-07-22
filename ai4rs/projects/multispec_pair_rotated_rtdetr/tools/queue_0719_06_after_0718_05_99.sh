#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/wangying01/lth/PairMOT/ai4rs
LAUNCHER=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/launch_0719_06_paper_base_longtail_reweight_99.sh
QUEUE_DIR=/data4/litianhao/PairMmot/workdir_99/0719_06_paper_base_longtail_reweight_r18_coco_full_1200x900_bf16_orderedpairs_fresh
LOCK=/data4/litianhao/PairMmot/workdir_99/.queue_0719_06.lock
PREDECESSOR=o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_adaptiveanchor_pcdp_coco_full_1200x900_bf16_99.py
MEMORY_LIMIT_MIB=2048

mkdir -p "${QUEUE_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] another 0719_06 queue is already active"
    exit 3
fi

echo "[$(date '+%F %T')] waiting for 0718_05 and GPUs 0,1"
ready_count=0
while true; do
    predecessor_active=0
    if pgrep -af "tools/train.py.*${PREDECESSOR}" >/dev/null; then
        predecessor_active=1
    fi

    mapfile -t used_mib < <(
        nvidia-smi --id=0,1 --query-gpu=memory.used \
            --format=csv,noheader,nounits | tr -d ' ')
    gpu_ready=0
    if [[ ${#used_mib[@]} -eq 2 ]] \
            && (( used_mib[0] < MEMORY_LIMIT_MIB )) \
            && (( used_mib[1] < MEMORY_LIMIT_MIB )); then
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
    echo "[$(date '+%F %T')] predecessor=${predecessor_active} gpu_mem=${used_mib[*]:-unavailable} MiB ready=${ready_count}/3"
    sleep 60
done

echo "[$(date '+%F %T')] conditions stable; starting 0719_06"
exec bash "${LAUNCHER}"
