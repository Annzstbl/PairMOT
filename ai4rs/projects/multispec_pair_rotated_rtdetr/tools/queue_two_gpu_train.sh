#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 CONFIG WORK_DIR GPUS PORT CONDA_ENV [MEM_LIMIT_MB] [CHECK_INTERVAL_SEC]" >&2
  exit 2
fi

CONFIG="$1"
WORK_DIR="$2"
GPUS="$3"
PORT="$4"
CONDA_ENV="$5"
MEM_LIMIT_MB="${6:-1024}"
CHECK_INTERVAL_SEC="${7:-300}"

REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
LOCK_DIR="${WORK_DIR}/queue.lock"
QUEUE_LOG="${WORK_DIR}/queue.log"
LAUNCH_LOG="${WORK_DIR}/launch.log"

mkdir -p "$WORK_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[$(date '+%F %T')] queue already active: $LOCK_DIR" | tee -a "$QUEUE_LOG"
  exit 1
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"
}

gpu_is_free() {
  local gpu="$1"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
  [ "$used" -le "$MEM_LIMIT_MB" ]
}

all_gpus_free() {
  local IFS=','
  local gpu
  read -ra gpu_ids <<< "$GPUS"
  for gpu in "${gpu_ids[@]}"; do
    if ! gpu_is_free "$gpu"; then
      return 1
    fi
  done
  return 0
}

activate_conda() {
  if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "Cannot find conda.sh under $HOME/anaconda3 or $HOME/miniconda3" >&2
    exit 3
  fi
  conda activate "$CONDA_ENV"
}

log "queued config=$CONFIG work_dir=$WORK_DIR gpus=$GPUS port=$PORT mem_limit=${MEM_LIMIT_MB}MB"
cd "$REPO_DIR"
activate_conda
export PYTHONPATH="$REPO_DIR"

while true; do
  if all_gpus_free; then
    log "GPUs $GPUS are free; launching training"
    export CUDA_VISIBLE_DEVICES="$GPUS"
    export PORT="$PORT"
    set +e
    bash tools/dist_train.sh "$CONFIG" 2 --work-dir "$WORK_DIR" >> "$LAUNCH_LOG" 2>&1
    status="$?"
    set -e
    log "training exited with status=$status"
    exit "$status"
  fi
  log "waiting for GPUs $GPUS"
  sleep "$CHECK_INTERVAL_SEC"
done
