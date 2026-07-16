#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/wangying01/lth/PairMOT/ai4rs
QUEUE_DIR=/data4/litianhao/PairMmot/workdir_99/0715_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_pairbandcontext
QUEUE_LOG=${QUEUE_DIR}/queue.log
PREV_CONFIG=o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_pairtransport_99.py

mkdir -p "${QUEUE_DIR}"
echo "[$(date '+%F %T')] queue started; waiting for 0715_02" >> "${QUEUE_LOG}"

while pgrep -f "tools/train.py.*${PREV_CONFIG}" >/dev/null; do
    sleep 60
done

while true; do
    mapfile -t gpu_mem < <(nvidia-smi \
        --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( gpu_mem[2] < 1000 && gpu_mem[3] < 1000 )); then
        break
    fi
    echo "[$(date '+%F %T')] GPUs 2,3 still occupied: ${gpu_mem[2]}, ${gpu_mem[3]} MiB" >> "${QUEUE_LOG}"
    sleep 60
done

echo "[$(date '+%F %T')] 0715_02 exited and GPUs are free; launching 0715_03" >> "${QUEUE_LOG}"
exec "${REPO}/projects/multispec_pair_rotated_rtdetr/tools/launch_0715_03_pairbandcontext_99.sh"
