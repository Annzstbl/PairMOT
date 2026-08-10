#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/wangying01/lth/PairMOT_0810_08_warmup12_cosine2667_99/ai4rs/ai4rs}
SMOKE_LAUNCHER=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/launch_smoke_0810_08_product_tangent_warmup12_cosine2667_decoder_99.sh
FORMAL_LAUNCHER=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/launch_0810_08_product_tangent_warmup12_cosine2667_decoder_99.sh
SMOKE_WORK_DIR=/data4/litianhao/PairMmot/workdir_99/smoke_0810_08_product_tangent_warmup12_cosine2667_4iter
FORMAL_WORK_DIR=/data4/litianhao/PairMmot/workdir_99/0810_08_final_product_tangent_warmup12_cosine2667_72e_2xb4_fresh
QUEUE_LOG=/data4/litianhao/PairMmot/workdir_99/queue_0810_08_warmup12_cosine2667.log
SCREEN_NAME=pm081008formal

test -d "${REPO}"
test -x "$(command -v nvidia-smi)"
test -f "${SMOKE_LAUNCHER}"
test -f "${FORMAL_LAUNCHER}"
test ! -e "${SMOKE_WORK_DIR}"
test ! -e "${FORMAL_WORK_DIR}"
test ! -e "${QUEUE_LOG}"
touch "${QUEUE_LOG}"
trap 'status=$?; echo "[$(date "+%F %T")] queue 0810_08 failed: status=${status} command=${BASH_COMMAND}" >> "${QUEUE_LOG}"; exit "${status}"' ERR

log() {
    echo "[$(date '+%F %T')] $*" >> "${QUEUE_LOG}"
}

select_two_free_gpus() {
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F',' '$2 + 0 < 1024 && $3 + 0 < 10 {gsub(/ /, "", $1); print $1}' \
        | head -n 2 \
        | paste -sd, -
}

two_gpus_idle_now() {
    local pair=$1
    local first=${pair%%,*}
    local second=${pair##*,}
    local idle_count
    idle_count=$(nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F',' -v a="${first}" -v b="${second}" \
            'BEGIN {n=0} {gsub(/ /, "", $1); if (($1==a || $1==b) && $2+0<1024 && $3+0<10) n++} END {print n}')
    [[ "${idle_count}" = 2 ]]
}

log "queue started; waiting for any two GPUs below 1024 MiB and 10% utilization"
candidate=''
stable_checks=0
while (( stable_checks < 3 )); do
    current=$(select_two_free_gpus)
    if [[ "${current}" == *,* ]]; then
        if [[ "${current}" = "${candidate}" ]]; then
            stable_checks=$((stable_checks + 1))
        else
            candidate=${current}
            stable_checks=1
            log "candidate GPUs=${candidate}; stability check 1/3"
        fi
    else
        if (( stable_checks > 0 )); then
            log "candidate lost before three checks"
        fi
        candidate=''
        stable_checks=0
    fi
    if (( stable_checks < 3 )); then
        sleep 20
    fi
done

two_gpus_idle_now "${candidate}"
test ! -e "${SMOKE_WORK_DIR}"
test ! -e "${FORMAL_WORK_DIR}"
log "three consecutive checks passed on GPUs=${candidate}; starting exact DDP smoke"
PAIRMOT_CUDA_VISIBLE_DEVICES=${candidate} bash "${SMOKE_LAUNCHER}"
test -s "${SMOKE_WORK_DIR}/iter_4.pth"
log "smoke and finite-checkpoint gates passed on GPUs=${candidate}"

for attempt in $(seq 1 12); do
    if two_gpus_idle_now "${candidate}"; then
        break
    fi
    if [[ "${attempt}" = 12 ]]; then
        log "smoke GPUs did not return idle before formal launch"
        exit 4
    fi
    sleep 5
done

test ! -e "${FORMAL_WORK_DIR}"
if screen -ls | grep -q "[.]${SCREEN_NAME}"; then
    log "screen ${SCREEN_NAME} already exists"
    exit 5
fi
log "starting fresh formal on dynamic GPUs=${candidate}"
screen -dmS "${SCREEN_NAME}" env PAIRMOT_CUDA_VISIBLE_DEVICES="${candidate}" bash "${FORMAL_LAUNCHER}"

for attempt in $(seq 1 120); do
    if [[ -f "${FORMAL_WORK_DIR}/launch.log" ]] \
        && grep -q 'Epoch(train)' "${FORMAL_WORK_DIR}/launch.log"; then
        if grep -Eiq 'Traceback|CUDA out of memory|loss: (nan|inf)|grad_norm: (nan|inf)|NCCL error|unused parameter.*error' "${FORMAL_WORK_DIR}/launch.log"; then
            log "formal reached a log interval but fatal scan failed"
            exit 6
        fi
        log "formal normal training interval observed; external five-gate audit may register RUNNING"
        exit 0
    fi
    if ! screen -ls | grep -q "[.]${SCREEN_NAME}"; then
        log "formal screen exited before a normal training interval"
        exit 7
    fi
    sleep 10
done

log "formal did not reach a normal training interval within 1200 seconds"
exit 8
