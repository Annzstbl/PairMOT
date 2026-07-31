#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data/users/litianhao/PairMOT_sync_3cb888d/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_0731_27_terminal_diagonal_factorized_evidence_decoder_4iter_smoke_197.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_197/smoke_0731_27_terminal_diagonal_factorized_evidence_4iter
PYTHON_ROOT=/data/users/litianhao/anaconda3/envs/py310/bin

test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/litianhao/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG
echo "[$(date '+%F %T')] fresh 0731_27 real-data DDP smoke" >> "${WORK_DIR}/launch.log"
"${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29968 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
if grep -Eiq 'Traceback|CUDA out of memory|loss: (nan|inf)|grad_norm: (nan|inf)|NCCL error|unused parameter.*error' "${WORK_DIR}/launch.log"; then exit 3; fi
test -f "${WORK_DIR}/iter_4.pth"
"${PYTHON_ROOT}/python" projects/multispec_pair_rotated_rtdetr/tools/check_terminal_diagonal_factorized_evidence_checkpoint.py "${WORK_DIR}/iter_4.pth" >> "${WORK_DIR}/launch.log" 2>&1
