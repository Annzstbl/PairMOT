#!/usr/bin/env bash
set -euo pipefail

# Manual, non-queued controlled run.  Bridge20B is the first py310 run using
# the freshly compiled native Detectron2 CUDA ROIAlignRotated pooler.
exec "$(dirname "$0")/run_hsmot_overfit_controlled_223.sh" \
    yolo11l_diffusion_det_hsmot_overfit_bridge20b_d2pooler.py \
    yolo11l_diffusion_det_hsmot_overfit_bridge20b_d2pooler_b1_acc1_100e_gpu0_v1 \
    fp32 3jpg
