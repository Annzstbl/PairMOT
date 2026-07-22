#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/wangying01/lth/PairMOT/ai4rs
LAUNCHER=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/launch_0720_03_liquid_qc_dualmoment_99.sh
QUEUE_DIR=/data4/litianhao/PairMmot/workdir_99/0720_03_paper_liquid_diffproduct_qc_dualmoment_r18_coco_full_1200x900_bf16_orderedpairs_fresh
LOCK=/data4/litianhao/PairMmot/workdir_99/.queue_0720_03.lock
PREDECESSOR=o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_longtail_reweight_coco_full_1200x900_bf16_99.py
MEMORY_LIMIT_MIB=2048
SMOKE_CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_liquid_diffproduct_qc_dualmoment_4iter_smoke_99.py
SMOKE_DIR=/data4/litianhao/PairMmot/workdir_99/smoke_0720_03_qc_dualmoment_queue_preflight

mkdir -p "${QUEUE_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] another 0720_03 queue is already active"
    exit 3
fi

echo "[$(date '+%F %T')] waiting for 0719_06 and GPUs 0,1"
ready_count=0
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
        ((ready_count += 1))
    else
        ready_count=0
    fi
    if (( ready_count >= 3 )); then
        break
    fi
    echo "[$(date '+%F %T')] predecessor=${predecessor_active} gpu_mem=${used_mib[*]:-unavailable} MiB ready=${ready_count}/3"
    sleep 60
done

source /data/users/wangying01/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd "${REPO}"
if [[ -e "${SMOKE_DIR}/epoch_1.pth" ]]; then
    echo "[$(date '+%F %T')] verified dual-moment smoke already exists"
else
    if [[ -d "${SMOKE_DIR}" ]] && find "${SMOKE_DIR}" -mindepth 1 -print -quit | grep -q .; then
        echo "Refusing to reuse incomplete smoke directory ${SMOKE_DIR}" >&2
        exit 4
    fi
    mkdir -p "${SMOKE_DIR}"
    export CUDA_VISIBLE_DEVICES=0,1
    export PORT=29914
    export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
    echo "[$(date '+%F %T')] starting 4-iteration dual-moment preflight"
    bash tools/dist_train.sh "${SMOKE_CONFIG}" 2 \
        --work-dir "${SMOKE_DIR}" > "${SMOKE_DIR}/launch.log" 2>&1
    test -f "${SMOKE_DIR}/epoch_1.pth"
    echo "[$(date '+%F %T')] dual-moment preflight passed"
fi

echo "[$(date '+%F %T')] conditions stable; starting 0720_03"
exec bash "${LAUNCHER}"
