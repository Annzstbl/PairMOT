#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 TRAIN_PGID STATE_DIR LOG" >&2
    exit 2
fi

TRAIN_PGID=$1
STATE_DIR=$2
LOG=$3
THRESHOLD_C=90
INTERVAL_S=10

mkdir -p "${STATE_DIR}"
while kill -0 "${TRAIN_PGID}" 2>/dev/null; do
    while IFS=',' read -r raw_index raw_temp; do
        index=${raw_index//[[:space:]]/}
        temp=${raw_temp//[[:space:]]/}
        [[ "${index}" =~ ^[0-3]$ ]] || continue
        [[ "${temp}" =~ ^[0-9]+$ ]] || continue
        if (( temp >= THRESHOLD_C )); then
            snapshot=$(nvidia-smi \
                --query-gpu=index,temperature.gpu,power.draw,utilization.gpu \
                --format=csv,noheader,nounits)
            {
                echo "[$(date '+%F %T')] THERMAL PAUSE: GPU ${index} reached ${temp} C"
                echo "${snapshot}"
                echo "Resume only after inspection: kill -CONT -- -${TRAIN_PGID}"
            } >> "${LOG}"
            printf '%s\n' "${index},${temp},$(date '+%F %T')" \
                > "${STATE_DIR}/THERMAL_PAUSED"
            kill -STOP -- "-${TRAIN_PGID}"
            exit 0
        fi
    done < <(nvidia-smi --query-gpu=index,temperature.gpu \
        --format=csv,noheader,nounits)
    sleep "${INTERVAL_S}"
done
