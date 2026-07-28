#!/usr/bin/env bash
set -euo pipefail

# Manual, non-queued controlled run: goal-time BS1 plus degree core only.
exec "$(dirname "$0")/run_hsmot_overfit_controlled_223.sh" \
    yolo11l_diffusion_det_hsmot_overfit_degreecore_minimal.py \
    yolo11l_diffusion_det_hsmot_overfit_degreecore_minimal_b1_acc1_100e_gpu0_v1 \
    fp32 3jpg
