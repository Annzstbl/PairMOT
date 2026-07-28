#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
LAUNCHER=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/launch_0727_06_encoder_sharedscalar_178.sh
PREDECESSOR_CONFIG=o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178.py
PREDECESSOR_DIR=/data4/litianhao/PairMmot/workdir_178/0727_01_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh
LOCK=/data4/litianhao/PairMmot/workdir_178/.queue_0727_06.lock
QUEUE_LOG=/data4/litianhao/PairMmot/workdir_178/queue_0727_06.log
SMOKE_CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_base_liquid_encoder_p5temporal_sharedscalar_pairdn_paircoherent_le180_4iter_smoke_178.py
SMOKE_DIR=/data4/litianhao/PairMmot/workdir_178/smoke_0727_06_base_liquid_encoder_p5temporal_sharedscalar_4iter
MEMORY_LIMIT_MIB=2048

exec >> "${QUEUE_LOG}" 2>&1
trap 'rc=$?; echo "[$(date '+"'"'%F %T'"'"')] FAILED_TO_START rc=${rc} command=${BASH_COMMAND}"; exit ${rc}' ERR
exec 9>"${LOCK}"
flock -n 9 || { echo "another 0727_06 queue is active"; exit 3; }

ready_count=0
while true; do
    predecessor_active=0
    pgrep -af "tools/train.py.*${PREDECESSOR_CONFIG}" >/dev/null \
        && predecessor_active=1
    checkpoint_ready=0
    [[ -f "${PREDECESSOR_DIR}/epoch_72.pth" ]] && checkpoint_ready=1
    if [[ -d "${PREDECESSOR_DIR}/val_track_eval" ]]; then
        eval_count=$(find "${PREDECESSOR_DIR}/val_track_eval" \
            -type d -name eval | wc -l)
    else
        eval_count=0
    fi
    used_mib=$(nvidia-smi --id=0 --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' ')
    if (( predecessor_active == 0 && checkpoint_ready == 1
            && eval_count >= 18 && used_mib < MEMORY_LIMIT_MIB )); then
        ((ready_count += 1))
    else
        ready_count=0
    fi
    (( ready_count >= 3 )) && break
    echo "[$(date '+%F %T')] predecessor=${predecessor_active} ckpt72=${checkpoint_ready} eval=${eval_count}/18 gpu=${used_mib:-unavailable}MiB ready=${ready_count}/3"
    sleep 60
done

set +u
source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
cd "${REPO}"
export PAIRMOT_HSMOT_ROOT=/data1/users/litianhao01/data/hsmot
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ ! -f "${SMOKE_DIR}/iter_4.pth" ]]; then
    [[ ! -e "${SMOKE_DIR}" ]] || {
        echo "Refusing incomplete smoke ${SMOKE_DIR}"
        exit 4
    }
    mkdir -p "${SMOKE_DIR}"
    python tools/train.py "${SMOKE_CONFIG}" \
        --work-dir "${SMOKE_DIR}" > "${SMOKE_DIR}/launch.log" 2>&1
    test -f "${SMOKE_DIR}/iter_4.pth"
fi

echo "[$(date '+%F %T')] smoke passed; launching formal 0727_06"
exec bash "${LAUNCHER}"
