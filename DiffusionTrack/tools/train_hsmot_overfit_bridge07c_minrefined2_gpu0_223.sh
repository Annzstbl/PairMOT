#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_hsmot_overfit_controlled_223.sh" \
    yolo11l_diffusion_det_hsmot_overfit_bridge07c_minrefined2.py \
    yolo11l_diffusion_det_hsmot_overfit_bridge07c_minrefined2_b1_acc1_100e_gpu0_v1 \
    bf16 3jpg
