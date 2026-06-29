#!/usr/bin/env bash
# Persist a lightweight 30-minute supervision trail for the formal pair run.
set -u

work_dir=${1:?usage: monitor_pair_half_training.sh WORK_DIR [INTERVAL_SEC]}
interval_sec=${2:-1800}
log_file="$work_dir/supervision.log"
launch_log="$work_dir/launch.log"

while true; do
  {
    date '+[%F %T %Z]'
    echo 'processes:'
    pgrep -af 'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn.py' || true
    echo 'gpus:'
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader || true
    echo 'latest-log:'
    tail -n 12 "$launch_log" 2>/dev/null || true
    echo
  } >> "$log_file"
  sleep "$interval_sec"
done
