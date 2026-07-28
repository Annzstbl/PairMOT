#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/litianhao01/PairMmot/ai4rs
ROOT=/data4/litianhao/PairMmot/workdir_252
PREDECESSOR=${ROOT}/0723_02_paper_liquid_independent_diffproduct_pairdn_independent_le180_r18_coco_full_1200x900_bf16_orderedpairs_fresh
SMOKE_DIR=${ROOT}/smoke_0723_07_pairdn_pecg_4iter
SMOKE_CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_pairdn_paircoherent_le180_pecg_4iter_smoke_252.py
FORMAL_LAUNCHER=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/launch_0723_07_pairdn_pecg_252.sh
LOG=${ROOT}/queue_0723_07_after_0723_02.log

trap 'status=$?; echo "[$(date "+%F %T")] ERROR status=${status} command=${BASH_COMMAND}" >> "${LOG}"; exit "${status}"' ERR
echo "[$(date '+%F %T')] queue started" >> "${LOG}"

while true; do
    eval_count=$(find "${PREDECESSOR}/val_track_eval" -name metrics.json -type f \
        -exec grep -l '"track/async_done": 1.0' {} + 2>/dev/null | wc -l)
    if ! pgrep -af 'tools/train.py.*0723_02' >/dev/null \
            && test -f "${PREDECESSOR}/epoch_72.pth" \
            && test "${eval_count}" -eq 18; then
        break
    fi
    echo "[$(date '+%F %T')] waiting: epoch72=$(test -f "${PREDECESSOR}/epoch_72.pth" && echo yes || echo no) eval=${eval_count}/18" >> "${LOG}"
    sleep 300
done

idle_checks=0
while test "${idle_checks}" -lt 3; do
    used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
        -i 0,1 | awk '{sum += $1} END {print sum + 0}')
    if test "${used}" -lt 1024; then
        idle_checks=$((idle_checks + 1))
    else
        idle_checks=0
    fi
    echo "[$(date '+%F %T')] gpu idle check ${idle_checks}/3 used=${used} MiB" >> "${LOG}"
    sleep 60
done

if test -e "${SMOKE_DIR}"; then
    echo "Smoke directory already exists: ${SMOKE_DIR}" >&2
    exit 3
fi

set +u
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=0,1
export PORT=29930
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] starting exact DDP smoke" >> "${LOG}"
bash tools/dist_train.sh "${SMOKE_CONFIG}" 2 --work-dir "${SMOKE_DIR}" >> "${LOG}" 2>&1
test -f "${SMOKE_DIR}/epoch_1.pth"
if grep -Eiq 'nan|traceback|out of memory|unused parameter|expected to have finished reduction' "${LOG}"; then
    echo "Smoke log contains a failure signature" >&2
    exit 4
fi

echo "[$(date '+%F %T')] smoke passed; launching formal experiment" >> "${LOG}"
screen -dmS train_0723_07_252 bash "${FORMAL_LAUNCHER}"
sleep 180
pgrep -af 'tools/train.py.*0723_07' >> "${LOG}"
echo "[$(date '+%F %T')] formal process observed; five-gate verification remains required" >> "${LOG}"
