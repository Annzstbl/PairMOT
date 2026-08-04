#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/litianhao/PairMOT_positiontransport_0803_28_197/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_28_iterative_cls_terminal_position_tangent_transport_decoder_197.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_197/0803_28_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_iterativeclsdnisolatede2e_terminalpositiontangenttransport_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh

set +u
source /data/users/litianhao/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data/users/litianhao/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin
trap 'status=$?; echo "[$(date "+%F %T")] formal 0803_28 on 197 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/litianhao/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data/users/litianhao/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1
: "${PAIRMOT_CUDA_VISIBLE_DEVICES:?set two currently free 197 GPU indices}"
export CUDA_VISIBLE_DEVICES=${PAIRMOT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh formal 0803_28 GPUs=${CUDA_VISIBLE_DEVICES} commit $(git rev-parse --short HEAD)" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/torchrun" --nproc_per_node=2 --master_port=29991 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    >> "${WORK_DIR}/launch.log" 2>&1
