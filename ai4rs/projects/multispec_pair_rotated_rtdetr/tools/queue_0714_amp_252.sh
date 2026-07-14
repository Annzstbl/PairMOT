#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=/data4/litianhao/PairMmot/workdir_252/0714_01_0704_resume_coco365_full_unique_allgt_amp
REPO=/data/users/litianhao01/PairMmot/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_amp_252.py
QUEUE_LOG=${WORK_DIR}/queue_amp.log
RUN_LOG=${WORK_DIR}/launch_amp.log
GPUS=0,1
PORT=29820

mkdir -p "${WORK_DIR}"
echo "[$(date '+%F %T')] queue start: ${CONFIG}, GPUs=${GPUS}, port=${PORT}" >> "${QUEUE_LOG}"

while true; do
  if pgrep -af "${WORK_DIR}|coco365_full_amp_252.py" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] target training already running; exit queue" >> "${QUEUE_LOG}"
    exit 0
  fi

  mem0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  mem1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
  if [[ "${mem0}" -lt 1024 && "${mem1}" -lt 1024 ]]; then
    echo "[$(date '+%F %T')] GPUs free: mem0=${mem0}MiB mem1=${mem1}MiB; launching" >> "${QUEUE_LOG}"
    break
  fi

  echo "[$(date '+%F %T')] wait: mem0=${mem0}MiB mem1=${mem1}MiB" >> "${QUEUE_LOG}"
  sleep 300
done

source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PORT
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] launch command begins" >> "${QUEUE_LOG}"
bash tools/dist_train.sh "${CONFIG}" 2 --work-dir "${WORK_DIR}" >> "${RUN_LOG}" 2>&1
status=$?
echo "[$(date '+%F %T')] launch command exits with status=${status}" >> "${QUEUE_LOG}"
exit "${status}"
