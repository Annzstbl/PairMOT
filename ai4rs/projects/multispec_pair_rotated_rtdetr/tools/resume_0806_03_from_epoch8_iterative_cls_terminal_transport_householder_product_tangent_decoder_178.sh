#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data1/users/litianhao01/PairMOT_householder_resume_0806_03_178/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_12e_hsmot_0806_03_iterative_cls_terminal_transport_householder_product_tangent_decoder_resume178.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0806_03_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_iterativeclsdnisolatede2e_pairsharedterminaltransporthouseholder_producttangentrefinement_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_resume_from_e8_to_e12
CHECKPOINT=/data4/litianhao/PairMmot/workdir_197/0804_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_iterativeclsdnisolatede2e_pairsharedterminaltransporthouseholder_producttangentrefinement_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh/epoch_8.pth

set +u
source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data1/users/litianhao01/anaconda3/envs/py310
PYTHON_ROOT=${CONDA_PREFIX}/bin
trap 'status=$?; echo "[$(date "+%F %T")] resume 0806_03 on 178 failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

test ! -e "${WORK_DIR}"
test -s "${CHECKPOINT}"
mkdir -p "${WORK_DIR}"
cd "${REPO}"
test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1
: "${PAIRMOT_CUDA_VISIBLE_DEVICES:?set one currently free 178 GPU index}"
case "${PAIRMOT_CUDA_VISIBLE_DEVICES}" in
    *','*) echo "0806_03 formal requires exactly one GPU" >&2; exit 2 ;;
esac
if nvidia-smi -i "${PAIRMOT_CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
    echo "selected GPU is not empty" >&2
    exit 3
fi
if pgrep -af "tools/train.py.*${WORK_DIR}" | grep -v 'pgrep -af'; then
    echo "matching training process already exists" >&2
    exit 4
fi
export CUDA_VISIBLE_DEVICES=${PAIRMOT_CUDA_VISIBLE_DEVICES}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] exact 0806_03 resume GPU=${CUDA_VISIBLE_DEVICES} from ${CHECKPOINT} commit $(git rev-parse --short HEAD)" >> "${WORK_DIR}/launch.log"
exec "${PYTHON_ROOT}/python" tools/train.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" --resume "${CHECKPOINT}" \
    >> "${WORK_DIR}/launch.log" 2>&1
