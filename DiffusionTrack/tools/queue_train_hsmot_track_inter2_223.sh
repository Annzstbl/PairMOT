#!/usr/bin/env bash
set -euo pipefail

# Do not race other users for a 24-GiB card. Stage-2 BS=1/rank requires
# roughly 12.5 GiB, so start only after both requested cards are free.
MAX_USED_MIB="${MAX_USED_MIB:-2048}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=',' read -r gpu_a_id gpu_b_id extra_gpu_id <<< "${GPU_IDS}"
if [[ -z "${gpu_a_id:-}" || -z "${gpu_b_id:-}" || -n "${extra_gpu_id:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly two physical GPU indices" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${gpu_a_id},${gpu_b_id}"

while true; do
  mapfile -t used < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  gpu_a="${used[gpu_a_id]// /}"
  gpu_b="${used[gpu_b_id]// /}"
  if (( gpu_a <= MAX_USED_MIB && gpu_b <= MAX_USED_MIB )); then
    break
  fi
  printf '%(%F %T)T waiting: GPU%s=%s MiB GPU%s=%s MiB\n' \
    -1 "${gpu_a_id}" "${gpu_a}" "${gpu_b_id}" "${gpu_b}"
  sleep "${POLL_SECONDS}"
done

exec "$(dirname "$0")/train_hsmot_track_inter2_223.sh"
