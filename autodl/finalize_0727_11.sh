#!/usr/bin/env bash
set -Eeuo pipefail

export WORK_DIR=/root/autodl-tmp/work_dirs/0727_11_paper_base_liquid_encoder_p5temporal_momentcompetitive_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_orderedpairs_autodl_1xb8_fresh
export EXPERIMENT_ID=0727_11
export EXPERIMENT_NAME='Paper Base + Liquid + MCDE Encoder'
export LAUNCHER_PID_FILE="$WORK_DIR/launcher.pid"
export FS_RESULT_ROOT=/root/autodl-fs/PairMOT_results/0727_11
export BASELINE_JSON=/root/autodl-fs/PairMOT_results/baselines/0716_02.json
export PYTHON_BIN=/root/miniconda3/bin/python3.12

exec /root/PairMOT/autodl/finalize_and_shutdown.sh
