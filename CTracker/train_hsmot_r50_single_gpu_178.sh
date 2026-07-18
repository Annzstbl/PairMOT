#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
CONDA_BIN="${CONDA_BIN:-/data1/users/litianhao01/anaconda3/bin/conda}"
RUN_NAME=ctracker_hsmot_r50_3dse_rotated_1200x900_bs4_acc2
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/data1/users/litianhao01/PairMOT/workdir_178/${RUN_NAME}}"
ARCHIVE_MODEL_DIR="${ARCHIVE_MODEL_DIR:-/data4/litianhao/PairMmot/workdir_178/${RUN_NAME}}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
DONE_FILE="${LOCAL_MODEL_DIR}/.training_done"

mkdir -p "${LOCAL_MODEL_DIR}"
rm -f "${DONE_FILE}"

bash tools/archive_checkpoints_178.sh \
  "${LOCAL_MODEL_DIR}" "${ARCHIVE_MODEL_DIR}" "${DONE_FILE}" \
  >"${LOCAL_MODEL_DIR}/archive.log" 2>&1 &
archive_pid=$!

finish_archive() {
  touch "${DONE_FILE}"
  wait "${archive_pid}" || true
}
trap finish_archive EXIT

train_args=(
  --dataset hsmot
  --root_path ../data/hsmot/train
  --ann_subdir mot
  --img_subdir npy2jpg
  --img_format 3jpg
  --image_scale 900 1200
  --depth 50
  --epochs 100
  --batch_size 4
  --accumulation_steps 2
  --workers 32
  --device cuda:0
  --lr 5e-5
  --stem_lr_multiplier 10
  --checkpoint_interval 1
  --model_dir "${LOCAL_MODEL_DIR}"
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  train_args+=(--no_pretrained --resume "${RESUME_CHECKPOINT}")
else
  train_args+=(--pretrained_model /data4/litianhao/PairMmot/pretrained_weights/ctracker_model_final.pt)
fi

# RTX 5090 single-GPU profile: micro-batch 4 x accumulation 2 preserves the
# original CTracker effective batch size of 8.
"${CONDA_BIN}" run --no-capture-output -n py310 python -u train.py \
  "${train_args[@]}" >"${LOCAL_MODEL_DIR}/train.log" 2>&1
