#!/usr/bin/env bash
set -euo pipefail

# Do not race other users for a 24-GiB card. Stage-2 BS=1/rank requires
# roughly 12.5 GiB, so start only after both requested cards are free.
MAX_USED_MIB="${MAX_USED_MIB:-2048}"
POLL_SECONDS="${POLL_SECONDS:-60}"

while true; do
  mapfile -t used < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  gpu2="${used[2]// /}"
  gpu3="${used[3]// /}"
  if (( gpu2 <= MAX_USED_MIB && gpu3 <= MAX_USED_MIB )); then
    break
  fi
  printf '%(%F %T)T waiting: GPU2=%s MiB GPU3=%s MiB\n' \
    -1 "${gpu2}" "${gpu3}"
  sleep "${POLL_SECONDS}"
done

exec "$(dirname "$0")/train_hsmot_track_inter2_223.sh"
