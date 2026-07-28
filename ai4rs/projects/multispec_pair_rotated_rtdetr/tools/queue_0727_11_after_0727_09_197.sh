#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/litianhao/PairMOT/ai4rs
LAUNCHER=${REPO}/projects/multispec_pair_rotated_rtdetr/tools/launch_0727_11_encoder_momentcompetitive_197.sh
PREDECESSOR_CONFIG=o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_detailspatial_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197.py
PREDECESSOR_DIR=/data4/litianhao/PairMmot/workdir_197/0727_09_paper_base_liquid_encoder_p5temporal_detailspatial_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_orderedpairs_fresh
LOCK=/data4/litianhao/PairMmot/workdir_197/.queue_0727_11.lock
QUEUE_LOG=/data4/litianhao/PairMmot/workdir_197/queue_0727_11.log
SMOKE_CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_base_liquid_encoder_p5temporal_momentcompetitive_pairdn_paircoherent_le180_4iter_smoke_197.py
SMOKE_DIR=/data4/litianhao/PairMmot/workdir_197/smoke_0727_11_base_liquid_encoder_p5temporal_momentcompetitive_4iter
MEMORY_LIMIT_MIB=2048

exec >> "${QUEUE_LOG}" 2>&1
trap 'rc=$?; echo "[$(date '+"'"'%F %T'"'"')] FAILED_TO_START rc=${rc} command=${BASH_COMMAND}"; exit ${rc}' ERR
exec 9>"${LOCK}"
flock -n 9 || { echo "another 0727_11 queue is active"; exit 3; }

ready_count=0
while true; do
    predecessor_active=0
    pgrep -af "tools/train.py.*${PREDECESSOR_CONFIG}" >/dev/null \
        && predecessor_active=1
    checkpoint_ready=0
    [[ -f "${PREDECESSOR_DIR}/epoch_72.pth" ]] && checkpoint_ready=1
    if [[ -d "${PREDECESSOR_DIR}/val_track_eval" ]]; then
        eval_count=$(find "${PREDECESSOR_DIR}/val_track_eval" \
            -mindepth 2 -maxdepth 2 -type f -name metrics.json | wc -l)
    else
        eval_count=0
    fi
    mapfile -t used_mib < <(
        nvidia-smi --id=4,5 --query-gpu=memory.used \
            --format=csv,noheader,nounits | tr -d ' ')
    if (( predecessor_active == 0 && checkpoint_ready == 1
            && eval_count >= 18 )) \
            && [[ ${#used_mib[@]} -eq 2 ]] \
            && (( used_mib[0] < MEMORY_LIMIT_MIB
                    && used_mib[1] < MEMORY_LIMIT_MIB )); then
        ((ready_count += 1))
    else
        ready_count=0
    fi
    (( ready_count >= 3 )) && break
    echo "[$(date '+%F %T')] predecessor=${predecessor_active} ckpt72=${checkpoint_ready} eval=${eval_count}/18 gpu=${used_mib[*]:-unavailable}MiB ready=${ready_count}/3"
    sleep 60
done

set +u
source /data/users/litianhao/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=4,5
export PORT=29942
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
if [[ ! -f "${SMOKE_DIR}/iter_4.pth" ]]; then
    [[ ! -e "${SMOKE_DIR}" ]] || {
        echo "Refusing incomplete smoke ${SMOKE_DIR}"
        exit 4
    }
    mkdir -p "${SMOKE_DIR}"
    bash tools/dist_train.sh "${SMOKE_CONFIG}" 2 \
        --work-dir "${SMOKE_DIR}" > "${SMOKE_DIR}/launch.log" 2>&1
    test -f "${SMOKE_DIR}/iter_4.pth"
    grep -Eq 'Epoch\(train\).*\[4/4\].*grad_norm:.*dn_loss_cls.*enc_loss_cls' \
        "${SMOKE_DIR}/launch.log"
    ! grep -Eiq 'Traceback|CUDA out of memory|NCCL|unused parameter|(^|[^a-z])(nan|inf)([^a-z]|$)' \
        "${SMOKE_DIR}/launch.log"
fi

echo "[$(date '+%F %T')] smoke passed; launching formal 0727_11"
exec bash "${LAUNCHER}"
