#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/litianhao01/PairMmot/ai4rs
WORK_DIR=/data4/litianhao/PairMmot/workdir_252/smoke_0729_03_dualevidence_decoder_dualoutputresidual_paircoherent_4iter
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_dualevidence_decoder_dualoutputresidual_paircoherent_4iter_smoke_252.py
LOG=${WORK_DIR}/launch.log

mkdir -p "${WORK_DIR}"
if find "${WORK_DIR}" -mindepth 1 -maxdepth 1 ! -name launch.log | grep -q .; then
    echo "Refusing a fresh smoke launch into non-empty ${WORK_DIR}" >&2
    exit 2
fi

cd "${REPO}"
grep -q 'dual_output_adapter' \
    projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_rotated_rtdetr_layers.py

set +u
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u

test -f /data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_test_gap1

export CUDA_VISIBLE_DEVICES=0,1
export PORT=29951
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0729_03 strict residual smoke" >> "${LOG}"
bash tools/dist_train.sh "${CONFIG}" 2 --work-dir "${WORK_DIR}" \
    >> "${LOG}" 2>&1

if grep -Eiq \
        'Traceback|CUDA out of memory|loss: (nan|inf)|grad_norm: (nan|inf)' \
        "${LOG}"; then
    echo "Smoke log contains a fatal or non-finite signal" >&2
    exit 3
fi

CHECKPOINT=${WORK_DIR}/iter_4.pth
test -f "${CHECKPOINT}"
python projects/multispec_pair_rotated_rtdetr/tools/verify_dual_output_adapter_smoke.py \
    "${CHECKPOINT}" >> "${LOG}" 2>&1
