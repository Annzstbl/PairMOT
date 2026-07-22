#!/usr/bin/env bash
set -Eeuo pipefail

source /root/PairMOT/autodl_runtime.env

WORK_DIR=/root/autodl-tmp/work_dirs/0718_02_paper_base_plus_liquid_anchorcompetitive_r18_coco_full_1200x900_bf16_orderedpairs_autodl_fresh
CONFIG=/root/PairMOT/ai4rs/projects/multispec_pair_rotated_rtdetr/configs/autodl_0718_02_paper_liquid_anchorcompetitive_full_1200x900_bf16.py

if [[ -e "$WORK_DIR" ]]; then
  echo "Refusing fresh launch into existing workdir: $WORK_DIR" >&2
  exit 2
fi
mkdir -p "$WORK_DIR"
echo "$$" > "$WORK_DIR/launcher.pid"

cd "$AI4RS_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export PORT=29891
export PYTHON=/root/miniconda3/bin/python
export PATH=/root/miniconda3/bin:$PATH
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0718_02 ARCR Liquid on GPUs $CUDA_VISIBLE_DEVICES" \
  >> "$WORK_DIR/launch.log"
exec bash tools/dist_train.sh "$CONFIG" 2 --work-dir "$WORK_DIR" \
  >> "$WORK_DIR/launch.log" 2>&1
