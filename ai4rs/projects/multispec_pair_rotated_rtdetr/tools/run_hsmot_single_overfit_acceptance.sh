#!/usr/bin/env bash
# Run single-frame HSMOT overfit acceptance in conda env py310.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

exec conda run --no-capture-output -n py310 python \
  projects/multispec_pair_rotated_rtdetr/tools/run_hsmot_single_overfit_acceptance.py \
  "$@"
