#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
LAUNCHER=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/launch_0721_01_liquid_qc_responsemass_178.sh
CACHE_SCRIPT=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/prepare_hsmot_shm_cache_178.sh
SOURCE_HSMOT_ROOT=/data1/users/litianhao01/data/hsmot
QUEUE_DIR=/data4/litianhao/PairMmot/workdir_178/0721_01_paper_liquid_diffproduct_qc_responsemass_r18_coco_full_1200x900_bf16_orderedpairs_1xb8_shm_fresh
LOCK=/data4/litianhao/PairMmot/workdir_178/.queue_0721_01.lock
PREDECESSOR=o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_rerun_coco_full_1200x900_bf16_178.py
MEMORY_LIMIT_MIB=2048
SMOKE_CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_liquid_diffproduct_qc_responsemass_4iter_smoke_178.py
SMOKE_DIR=/data4/litianhao/PairMmot/workdir_178/smoke_0721_01_qc_responsemass_shm_4iter

mkdir -p "${QUEUE_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] another 0721_01 queue is already active"
    exit 3
fi

echo "[$(date '+%F %T')] waiting for 0719_05 final evaluation and GPU 0"
ready_count=0
while true; do
    predecessor_active=0
    if pgrep -af "tools/train.py.*${PREDECESSOR}" >/dev/null; then
        predecessor_active=1
    fi
    used_mib=$(nvidia-smi --id=0 --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' ')
    gpu_ready=0
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
            && (( used_mib < MEMORY_LIMIT_MIB )); then
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
    echo "[$(date '+%F %T')] predecessor=${predecessor_active} gpu0_mem=${used_mib:-unavailable} MiB ready=${ready_count}/3"
    sleep 60
done

set +u
source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PAIRMOT_HSMOT_ROOT="${SOURCE_HSMOT_ROOT}"
if cache_root=$(bash "${CACHE_SCRIPT}"); then
    export PAIRMOT_HSMOT_ROOT="${cache_root}"
    echo "[$(date '+%F %T')] preflight uses validated tmpfs cache ${cache_root}"
else
    echo "[$(date '+%F %T')] cache preparation failed; preflight uses local NVMe fallback"
fi

if [[ -e "${SMOKE_DIR}/epoch_1.pth" ]]; then
    echo "[$(date '+%F %T')] verified response-mass single-GPU smoke already exists"
else
    if [[ -d "${SMOKE_DIR}" ]] \
            && find "${SMOKE_DIR}" -mindepth 1 -print -quit | grep -q .; then
        echo "Refusing to reuse incomplete smoke directory ${SMOKE_DIR}" >&2
        exit 4
    fi
    mkdir -p "${SMOKE_DIR}"
    echo "[$(date '+%F %T')] starting 4-iteration response-mass preflight"
    python tools/train.py "${SMOKE_CONFIG}" --work-dir "${SMOKE_DIR}" \
        > "${SMOKE_DIR}/launch.log" 2>&1
    test -f "${SMOKE_DIR}/epoch_1.pth"
    echo "[$(date '+%F %T')] response-mass preflight passed"
fi

echo "[$(date '+%F %T')] conditions stable; starting 0721_01"
exec bash "${LAUNCHER}"
