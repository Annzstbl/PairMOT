#!/usr/bin/env bash
set -eo pipefail

ROOT="/data/users/litianhao01/PairMmot"
cd "$ROOT"

export PS1=""
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310

GPU_IDS="${GPU_IDS:-2,3}"
MASTER_PORT="${MASTER_PORT:-29672}"
BASE_WORKDIR="workdir/o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_rgbrepeat2d"
SUMMARY="workdir/r18_followup_queue_summary.jsonl"
LOG="workdir/r18_followup_queue.log"

mkdir -p workdir
touch "$SUMMARY" "$LOG"

configs=(
  "ai4rs/projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_rgbrepeat2d_stemlr2x.py"
  "ai4rs/projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_rgbrepeat2d_bblr005.py"
  "ai4rs/projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_rgbrepeat2d_bblr02.py"
  "ai4rs/projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_interp2d.py"
  "ai4rs/projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_3dse_reduction2.py"
)

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

metric_json() {
  local wd="$1"
  python - "$wd" <<'PY'
import glob
import json
import os
import sys

wd = sys.argv[1]
best = None
latest = None
for path in sorted(glob.glob(os.path.join(wd, '*', 'vis_data', 'scalars.json'))):
    with open(path, 'r') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'hsmot/mAP' not in row:
                continue
            latest = row
            if best is None or row['hsmot/mAP'] > best['hsmot/mAP']:
                best = row
out = {
    'work_dir': wd,
    'best_mAP': None if best is None else best.get('hsmot/mAP'),
    'best_AP50': None if best is None else best.get('hsmot/AP50'),
    'best_step': None if best is None else best.get('step'),
    'latest_mAP': None if latest is None else latest.get('hsmot/mAP'),
    'latest_step': None if latest is None else latest.get('step'),
}
print(json.dumps(out, sort_keys=True))
PY
}

wait_for_rgbrepeat2d() {
  log "waiting for base rgbrepeat2d to finish: $BASE_WORKDIR"
  while true; do
    if [[ -f "$BASE_WORKDIR/epoch_72.pth" ]]; then
      log "base rgbrepeat2d finished"
      metric_json "$BASE_WORKDIR" | tee -a "$SUMMARY"
      return 0
    fi
    if ! pgrep -f "o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_rgbrepeat2d.py" >/dev/null; then
      log "base rgbrepeat2d process stopped before epoch_72.pth; aborting queue"
      metric_json "$BASE_WORKDIR" | tee -a "$SUMMARY" || true
      return 1
    fi
    sleep 300
  done
}

run_one() {
  local cfg="$1"
  local name
  name="$(basename "$cfg" .py)"
  local wd="workdir/$name"
  if [[ -f "$wd/epoch_72.pth" ]]; then
    log "skip completed: $name"
    metric_json "$wd" | tee -a "$SUMMARY"
    return 0
  fi
  log "start experiment: $name"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" PORT="$MASTER_PORT" bash ai4rs/tools/dist_train.sh "$cfg" 2 --work-dir "$wd" 2>&1 | tee -a "$LOG"
  local status=${PIPESTATUS[0]}
  metric_json "$wd" | tee -a "$SUMMARY"
  if [[ $status -ne 0 ]]; then
    log "experiment failed: $name status=$status"
    return "$status"
  fi
  log "finish experiment: $name"
}

wait_for_rgbrepeat2d

count=0
for cfg in "${configs[@]}"; do
  count=$((count + 1))
  if [[ "$count" -gt 5 ]]; then
    log "experiment limit reached: 5"
    break
  fi
  run_one "$cfg"
done

log "queue finished"
