#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/litianhao01/PairMmot/ai4rs
LAUNCHER=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/launch_0721_04_liquid_bsac_accuracyfix_252.sh
LOCK=/data4/litianhao/PairMmot/workdir_252/.queue_0721_04.lock
PREDECESSOR=o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_diffproduct_qc_gatemass_coco_full_1200x900_bf16_252.py
SMOKE_CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_liquid_bsac_accuracyfix_4iter_smoke_252.py
SMOKE_DIR=/data4/litianhao/PairMmot/workdir_252/smoke_0721_04_bsac_4iter
MEMORY_LIMIT_MIB=2048

exec 9>"${LOCK}"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] another 0721_04 queue is active"
    exit 3
fi

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
    echo "[$(date '+%F %T')] predecessor=${predecessor_active} gpu_mem=${used_mib[*]:-unavailable} ready=${ready_count}/3"
    sleep 60
done

set +u
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
cd "${REPO}"
if [[ ! -e "${SMOKE_DIR}/epoch_1.pth" ]]; then
    if [[ -d "${SMOKE_DIR}" ]] && find "${SMOKE_DIR}" -mindepth 1 -print -quit | grep -q .; then
        echo "Refusing incomplete smoke directory ${SMOKE_DIR}" >&2
        exit 4
    fi
    mkdir -p "${SMOKE_DIR}"
    export CUDA_VISIBLE_DEVICES=0,1
    export PORT=29920
    export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
    bash tools/dist_train.sh "${SMOKE_CONFIG}" 2 \
        --work-dir "${SMOKE_DIR}" > "${SMOKE_DIR}/launch.log" 2>&1
    test -f "${SMOKE_DIR}/epoch_1.pth"
fi

exec bash "${LAUNCHER}"
