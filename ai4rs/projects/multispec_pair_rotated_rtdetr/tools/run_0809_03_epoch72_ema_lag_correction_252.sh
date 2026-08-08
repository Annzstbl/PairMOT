#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/litianhao01/PairMOT_terminaltransportproduct_0804_01_resume252/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_resume252.py
ROOT=/data4/litianhao/PairMmot/workdir_252/0809_03_product_tangent_epoch72_ema_lag_correction_eval
VERIFY=${ROOT}/tools/verify_decoder_epoch72_goal.py

if pgrep -af '[t]ools/train.py.*0808_08.*adamclock' >/dev/null; then
    echo 'Refusing EMA-lag evaluation while 0808_08 training is active' >&2
    exit 2
fi

for gpu in 0 1; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -i "${gpu}" | tr -d ' ')
    if (( used >= 500 )); then
        echo "Refusing occupied GPU${gpu}: ${used} MiB" >&2
        exit 3
    fi
done
sleep 5
for gpu in 0 1; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -i "${gpu}" | tr -d ' ')
    if (( used >= 500 )); then
        echo "Refusing unstable GPU${gpu}: ${used} MiB" >&2
        exit 4
    fi
done

test -d "${REPO}/.git"
test -f "${REPO}/${CONFIG}"
test -f "${VERIFY}"

run_one() {
    local fraction=$1
    local gpu=$2
    local checkpoint=${ROOT}/checkpoints/online_fraction_${fraction}.pth
    local output=${ROOT}/online_fraction_${fraction}
    test -f "${checkpoint}"
    test ! -e "${output}"
    mkdir -p "${output}"
    ln -s "${checkpoint}" "${output}/epoch_72.pth"
    (
        set +u
        source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
        conda activate py310
        set -u
        cd "${REPO}"
        export CUDA_VISIBLE_DEVICES=${gpu}
        export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        python tools/test.py "${CONFIG}" "${checkpoint}" \
            --work-dir "${output}" \
            --cfg-options \
            "test_evaluator.metrics.track_eval_out_dir=${output}/val_track_eval" \
            "test_evaluator.metrics.val_det_out_dir=${output}/val_det" \
            test_evaluator.metrics.async_track_eval=True \
            > "${output}/test.log" 2>&1
    )
}

run_one 025 0 &
pid_025=$!
run_one 050 1 &
pid_050=$!
status=0
wait "${pid_025}" || status=1
wait "${pid_050}" || status=1
if (( status != 0 )); then
    echo 'At least one EMA-lag evaluation failed' >&2
    exit "${status}"
fi

for fraction in 025 050; do
    metrics=${ROOT}/online_fraction_${fraction}/val_track_eval/val_track_0001/metrics.json
    for _ in $(seq 1 120); do
        [[ -s "${metrics}" ]] && break
        sleep 15
    done
    test -s "${metrics}"
done

set +e
/data/users/litianhao01/anaconda3/envs/py310/bin/python "${VERIFY}" \
    "${ROOT}/online_fraction_025" --epoch 72 --payload-step 1 \
    --json-out "${ROOT}/online_fraction_025/strict_verification.json"
verify_025=$?
/data/users/litianhao01/anaconda3/envs/py310/bin/python "${VERIFY}" \
    "${ROOT}/online_fraction_050" --epoch 72 --payload-step 1 \
    --json-out "${ROOT}/online_fraction_050/strict_verification.json"
verify_050=$?
set -e
if (( verify_025 != 0 && verify_025 != 2 )); then
    exit "${verify_025}"
fi
if (( verify_050 != 0 && verify_050 != 2 )); then
    exit "${verify_050}"
fi
printf 'fraction025_verifier=%s\nfraction050_verifier=%s\n' \
    "${verify_025}" "${verify_050}"
