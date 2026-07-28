#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/linxu/code/DiffusionTrack-PairMOT
RUNTIME=/data4/linxu/PairMOT_DiffusionTrack
LX_RUNTIME=${RUNTIME}/lx_baseline_isolated
QUEUE_LOG=${RUNTIME}/logs/queue_lx_then_bs2_then_full30_gpu0_20260724.log
LX_LOG=${LX_RUNTIME}/logs/lx_baseline_diffusiontrack_single_data43_2_x20_gpu0_val2_thresh001_diag_v3.log
LX_METRICS=${LX_RUNTIME}/outputs/lx_baseline_diffusiontrack_single_data43_2_x20_gpu0_val2_thresh001_diag_v3/val_det
BS1_LOG=${RUNTIME}/logs/stage1_overfit_data43_2_legacycenter_covered_lxlr_b1_acc1_100e_gpu0_v2_20260723.log
BS2_LOG=${RUNTIME}/logs/stage1_overfit_data43_2_legacycenter_covered_lxlr_b2_acc1_100e_gpu0_v3_20260724.log
COMPARE1=${RUNTIME}/logs/compare_ours_bs1_vs_lx_20260724.json
COMPARE2=${RUNTIME}/logs/compare_ours_bs1_vs_bs2_20260724.json

exec >>"${QUEUE_LOG}" 2>&1
trap 'rc=$?; echo "$(date -Is) FAILED rc=${rc} command=${BASH_COMMAND}"' ERR

echo "$(date -Is) queue started"
while pgrep -f \
        "lx_baseline_diffusiontrack_single_data43_2_x20_gpu0_val2_thresh001_diag_v3" \
        >/dev/null; do
    if grep -Eq "Traceback|CUDA out of memory|Error nan" "${LX_LOG}"; then
        echo "$(date -Is) LX predecessor failed"
        exit 10
    fi
    sleep 30
done
grep -q "Training of experiment is done" "${LX_LOG}"
echo "$(date -Is) LX predecessor completed"

/data/users/linxu/.conda/envs/py310_pairmot/bin/python \
    "${REPO}/tools/compare_overfit_ap.py" \
    --ours-bs1-log "${BS1_LOG}" \
    --lx-metrics-root "${LX_METRICS}" \
    --output "${COMPARE1}"
echo "$(date -Is) ours-vs-LX gate passed"

for _ in 1 2 3; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        --id=0 | tr -d ' ')
    test "${used}" -lt 2048
    sleep 10
done

echo "$(date -Is) starting BS2/acc1 overfit"
bash "${REPO}/tools/train_hsmot_overfit_legacy_penalty_covered_b2_acc1_gpu0_223.sh"
grep -q "Training of experiment is done" "${BS2_LOG}"

/data/users/linxu/.conda/envs/py310_pairmot/bin/python \
    "${REPO}/tools/compare_overfit_ap.py" \
    --ours-bs1-log "${BS1_LOG}" \
    --lx-metrics-root "${LX_METRICS}" \
    --ours-bs2-log "${BS2_LOG}" \
    --output "${COMPARE2}"
echo "$(date -Is) BS2-slower gate passed"

for _ in 1 2 3; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        --id=0 | tr -d ' ')
    test "${used}" -lt 2048
    sleep 10
done

echo "$(date -Is) starting full-data 30-epoch BS1/acc1 training"
exec bash "${REPO}/tools/train_hsmot_stage1_full30_b1_acc1_gpu0_223.sh"
