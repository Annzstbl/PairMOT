#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0731_21_terminal_orthogonal_factorized_evidence_decoder_178.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0731_21_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalorthogonalfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh
CHECKPOINT=${WORK_DIR}/epoch_40.pth
PYTHON_ROOT=/data1/users/litianhao01/anaconda3/envs/py310/bin

trap 'status=$?; echo "[$(date "+%F %T")] resume 0731_21 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

cd "${REPO}"
test -s "${CHECKPOINT}"
test "$(cat "${WORK_DIR}/last_checkpoint")" = "${CHECKPOINT}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1
if pgrep -af "tools/train.py.*${WORK_DIR}" | grep -v 'pgrep -af'; then
    echo "matching training process already exists" >&2
    exit 2
fi
if nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU0 is not empty" >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "[$(date '+%F %T')] exact resume 0731_21 from epoch_40 commit $(git rev-parse --short HEAD)" \
    >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/python" tools/train.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" --resume "${CHECKPOINT}" \
    >> "${WORK_DIR}/launch.log" 2>&1
