#!/usr/bin/env bash
set -Eeuo pipefail

: "${WORK_DIR:?Set WORK_DIR}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID}"
: "${EXPERIMENT_NAME:?Set EXPERIMENT_NAME}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPECTED_EVALS="${EXPECTED_EVALS:-18}"
POLL_SECONDS="${POLL_SECONDS:-60}"
ASYNC_TIMEOUT_SECONDS="${ASYNC_TIMEOUT_SECONDS:-7200}"
LAUNCHER_PID_FILE="${LAUNCHER_PID_FILE:-$WORK_DIR/launcher.pid}"
FS_RESULT_ROOT="${FS_RESULT_ROOT:-/root/autodl-fs/PairMOT_results/$EXPERIMENT_ID}"
REPORT_REL="ai4rs/projects/multispec_pair_rotated_rtdetr/docs/reports/autodl/${EXPERIMENT_ID}_result.md"
JSON_REL="ai4rs/projects/multispec_pair_rotated_rtdetr/docs/reports/autodl/${EXPERIMENT_ID}_result.json"
GITHUB_REPO="${GITHUB_REPO:-git@github.com:Annzstbl/PairMOT.git}"
GITHUB_BRANCH="${GITHUB_BRANCH:-autodl/${EXPERIMENT_ID}-results-$(date +%Y%m%d)}"
DEPLOY_KEY="${DEPLOY_KEY:-/root/.ssh/pairmot_results_ed25519}"
BASELINE_JSON="${BASELINE_JSON:-/root/autodl-fs/PairMOT_results/baselines/0716_04.json}"
STATE_DIR="$FS_RESULT_ROOT/finalizer_state"
mkdir -p "$STATE_DIR"
exec >> "$STATE_DIR/finalizer.log" 2>&1

power_off() {
  rm -f "$DEPLOY_KEY" "$DEPLOY_KEY.pub" || true
  sync
  /usr/bin/shutdown
}

shutdown_after_error() {
  code=$?
  line=$1
  printf 'finalizer_error exit=%s line=%s artifacts_root=%s\n' \
    "$code" "$line" "$FS_RESULT_ROOT" > "$STATE_DIR/status"
  power_off
  exit "$code"
}
trap 'shutdown_after_error $LINENO' ERR

echo "[$(date '+%F %T')] finalizer started for $EXPERIMENT_ID"
launcher_pid=$(cat "$LAUNCHER_PID_FILE")
while kill -0 "$launcher_pid" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done
echo "[$(date '+%F %T')] training launcher exited"

if [[ ! -s "$WORK_DIR/epoch_72.pth" ]]; then
  echo "Training did not produce epoch_72.pth; preserving failure artifacts."
  mkdir -p "$FS_RESULT_ROOT/artifacts"
  cp -a "$WORK_DIR/launch.log" "$FS_RESULT_ROOT/artifacts/" || true
  printf 'training_failed\n' > "$STATE_DIR/status"
  power_off
  exit 1
fi

deadline=$((SECONDS + ASYNC_TIMEOUT_SECONDS))
while true; do
  completed=$(find "$WORK_DIR/val_track_eval" -mindepth 2 -maxdepth 2 \
    -type f -name metrics.json -print0 2>/dev/null | \
    xargs -0 -r grep -l '"track/async_done": 1.0' | wc -l)
  running=$(pgrep -fc 'async_pair_track_eval.py' || true)
  echo "[$(date '+%F %T')] async eval: completed=$completed/$EXPECTED_EVALS running=$running"
  if [[ "$completed" -eq "$EXPECTED_EVALS" && "$running" -eq 0 ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for asynchronous TrackEval; preserving artifacts."
    printf 'async_eval_timeout completed=%s expected=%s\n' \
      "$completed" "$EXPECTED_EVALS" > "$STATE_DIR/status"
    power_off
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

STAGE="$FS_RESULT_ROOT/stage"
mkdir -p "$STAGE"
baseline_args=()
[[ -f "$BASELINE_JSON" ]] && baseline_args=(--baseline "$BASELINE_JSON")
python "$SCRIPT_DIR/analyze_experiment.py" \
  --work-dir "$WORK_DIR" --experiment-id "$EXPERIMENT_ID" \
  --experiment-name "$EXPERIMENT_NAME" --expected-evals "$EXPECTED_EVALS" \
  --output-md "$STAGE/result.md" --output-json "$STAGE/result.json" \
  "${baseline_args[@]}"

best_epoch=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["best"]["epoch"])' \
  "$STAGE/result.json")
ARTIFACTS="$FS_RESULT_ROOT/artifacts"
mkdir -p "$ARTIFACTS"
cp -a "$STAGE/result.md" "$STAGE/result.json" "$WORK_DIR/launch.log" "$ARTIFACTS/"
find "$WORK_DIR" -maxdepth 2 -type f \( -name scalars.json -o -name '*.py' \) \
  -exec cp -a --parents {} "$ARTIFACTS" \;
cp -a "$WORK_DIR/epoch_${best_epoch}.pth" "$ARTIFACTS/" 
if [[ "$best_epoch" != 72 ]]; then
  cp -a "$WORK_DIR/epoch_72.pth" "$ARTIFACTS/"
fi
tar -C "$WORK_DIR" -czf "$ARTIFACTS/val_track_eval.tar.gz" val_track_eval
find "$ARTIFACTS" -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$FS_RESULT_ROOT/SHA256SUMS"
printf 'artifacts_preserved\n' > "$STATE_DIR/status"

PUBLISH_ROOT=/root/autodl-tmp/pairmot_result_publish
cloned=0
for attempt in $(seq 1 10); do
  rm -rf "$PUBLISH_ROOT"
  if GIT_SSH_COMMAND="ssh -i $DEPLOY_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
      git clone "$GITHUB_REPO" "$PUBLISH_ROOT"; then
    cloned=1
    break
  fi
  echo "GitHub clone attempt $attempt failed; retrying in 60 seconds."
  sleep 60
done
if [[ "$cloned" -ne 1 ]]; then
  printf 'publish_failed_clone artifacts_preserved best_epoch=%s\n' \
    "$best_epoch" > "$STATE_DIR/status"
  power_off
  exit 0
fi
git -C "$PUBLISH_ROOT" checkout -b "$GITHUB_BRANCH"
mkdir -p "$PUBLISH_ROOT/$(dirname -- "$REPORT_REL")"
cp "$STAGE/result.md" "$PUBLISH_ROOT/$REPORT_REL"
cp "$STAGE/result.json" "$PUBLISH_ROOT/$JSON_REL"
git -C "$PUBLISH_ROOT" add "$REPORT_REL" "$JSON_REL"
git -C "$PUBLISH_ROOT" -c user.name='PairMOT AutoDL' \
  -c user.email='autodl-results@users.noreply.github.com' \
  commit -m "docs: add $EXPERIMENT_ID AutoDL results"

published=0
for attempt in $(seq 1 10); do
  if GIT_SSH_COMMAND="ssh -i $DEPLOY_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
      git -C "$PUBLISH_ROOT" push -u origin "$GITHUB_BRANCH"; then
    published=1
    break
  fi
  echo "GitHub push attempt $attempt failed; retrying in 60 seconds."
  sleep 60
done
if [[ "$published" -eq 1 ]]; then
  printf 'published branch=%s best_epoch=%s\n' "$GITHUB_BRANCH" "$best_epoch" \
    > "$STATE_DIR/status"
else
  printf 'publish_failed artifacts_preserved best_epoch=%s\n' "$best_epoch" \
    > "$STATE_DIR/status"
fi
echo "[$(date '+%F %T')] finalization complete; shutting down AutoDL"
power_off
