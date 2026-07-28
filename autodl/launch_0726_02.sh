#!/usr/bin/env bash
set -Eeuo pipefail

source /root/PairMOT/autodl_runtime.env

WORK_DIR=/root/autodl-tmp/work_dirs/0726_02_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_orderedpairs_autodl_1xb8_fresh
CONFIG=/root/PairMOT/ai4rs/projects/multispec_pair_rotated_rtdetr/configs/autodl_0726_02_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_full_1200x900_bf16_1xb8.py
PYTHON_BIN=/root/miniconda3/bin/python3.12

if [[ -e "$WORK_DIR" ]]; then
  echo "Refusing fresh launch into existing workdir: $WORK_DIR" >&2
  exit 2
fi
mkdir -p "$WORK_DIR"
echo "$$" > "$WORK_DIR/launcher.pid"

cd "$AI4RS_ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHON="$PYTHON_BIN"
export PATH=/root/miniconda3/bin:$PATH
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0726_02 on one RTX 5090, physical batch 8" \
  >> "$WORK_DIR/launch.log"
exec "$PYTHON_BIN" tools/train.py "$CONFIG" --work-dir "$WORK_DIR" \
  >> "$WORK_DIR/launch.log" 2>&1
