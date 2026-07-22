#!/usr/bin/env bash
set -euo pipefail

LAUNCHER=/data/users/wangying01/lth/PairMOT/ai4rs/projects/multispec_pair_rotated_rtdetr/tools/launch_0718_03_paper_liquid_anchorcompetitive_adaptiveanchor_99.sh
LOCK=/data4/litianhao/PairMmot/workdir_99/.queue_0718_03.lock
PREDECESSOR=o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_originalhard_coco_full_1200x900_bf16_99.py
MEMORY_LIMIT_MIB=1024

exec 9>"${LOCK}"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] another 0718_03 queue is already active"
    exit 3
fi

echo "[$(date '+%F %T')] waiting for 0717_02 and GPUs 0,1"
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
        break
    fi
    echo "[$(date '+%F %T')] predecessor=${predecessor_active} gpu_mem=${used_mib[*]:-unavailable} MiB"
    sleep 60
done

echo "[$(date '+%F %T')] conditions satisfied; starting 0718_03"
exec bash "${LAUNCHER}"
