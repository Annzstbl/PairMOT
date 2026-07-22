#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/linxu/code/DiffusionTrack-PairMOT
ENV=/data/users/linxu/.conda/envs/py310_pairmot
RUNTIME=/data4/linxu/PairMOT_DiffusionTrack
EXP_NAME=yolo11l_diffusion_det_hsmot_nomosaic_fixed896x1184_b4_d2_acc1_bf16_plainriou_v4
LOG=${RUNTIME}/logs/stage1_det_nomosaic_fixed896x1184_b4_d2_acc1_bf16_plainriou_v4_20260722.log
QUEUE_LOG=${RUNTIME}/logs/queue_stage1_det_nomosaic_fixed896x1184_gpu01_v4_20260722.log

export PATH="${ENV}/bin:/data/users/linxu/anaconda3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HSMOT_TRAIN_ROOT=/data/users/qinhaolin01/data/hsmot/train
export HSMOT_VAL_ROOT=/data/users/qinhaolin01/data/hsmot/test
export HSMOT_IMG_SUBDIR=npy2jpg
export HSMOT_IMG_FORMAT=3jpg
export YOLO11_WEIGHTS=${RUNTIME}/pretrained_weights/mmot_official/yolo11L-8ch-3dstem.pt

mkdir -p "${RUNTIME}/logs"
exec 9>"${RUNTIME}/queue_stage1_det_nomosaic_gpu01.lock"
if ! flock -n 9; then
    echo "another Stage-1 no-Mosaic GPU 0/1 queue is already active" | tee -a "${QUEUE_LOG}"
    exit 1
fi

echo "$(date '+%F %T %Z') waiting for physical GPU 0 and GPU 1" | tee -a "${QUEUE_LOG}"
stable=0
checks=0
while (( stable < 3 )); do
    mapfile -t used < <(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( ${used[0]} < 1024 && ${used[1]} < 1024 )); then
        stable=$((stable + 1))
    else
        stable=0
    fi
    checks=$((checks + 1))
    if (( checks == 1 || checks % 10 == 0 )); then
        echo "$(date '+%F %T %Z') gpu0=${used[0]}MiB gpu1=${used[1]}MiB stable=${stable}/3" | tee -a "${QUEUE_LOG}"
    fi
    if (( stable < 3 )); then
        sleep 30
    fi
done

echo "$(date '+%F %T %Z') GPU 0/1 available; starting ${EXP_NAME}" | tee -a "${QUEUE_LOG}"
cd "${REPO}"
exec "${ENV}/bin/python" tools/train.py \
    -f exps/example/mot/yolo11l_diffusion_det_hsmot_nomosaic_fixed896x1184.py \
    -d 2 -b 4 --accumulate 1 --amp-dtype bf16 \
    --experiment-name "${EXP_NAME}" \
    output_dir "${RUNTIME}/work_dirs" \
    >"${LOG}" 2>&1
