#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0731_21_terminal_orthogonal_factorized_evidence_decoder_178.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0731_21_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalorthogonalfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh
PYTHON_ROOT=/data1/users/litianhao01/anaconda3/envs/py310/bin

cd "${REPO}"
test -s "${WORK_DIR}/epoch_32.pth"
test "$(cat "${WORK_DIR}/last_checkpoint")" = "${WORK_DIR}/epoch_32.pth"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG
# PyTorch 2.6 defaults torch.load() to weights_only=True. MMEngine resume
# restores trusted optimizer/scheduler/HistoryBuffer objects from our own
# formal checkpoint, so opt out only for this exact-resume process.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "[$(date '+%F %T')] exact resume 0731_21 from epoch_32 commit $(git rev-parse --short HEAD)" \
    >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/python" tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" --resume \
    >> "${WORK_DIR}/launch.log" 2>&1
