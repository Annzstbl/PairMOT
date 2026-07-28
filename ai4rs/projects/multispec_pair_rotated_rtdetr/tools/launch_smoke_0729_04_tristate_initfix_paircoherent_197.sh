#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/litianhao/PairMOT/ai4rs
WORK_DIR=/data4/litianhao/PairMmot/workdir_197/smoke_0729_04_base_liquid_encoder_dualevidence_decoder0708_03_initfix_pairdn_paircoherent_4iter
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_base_liquid_encoder_p5temporal_dualevidence_decoder0708_03_initfix_pairdn_paircoherent_4iter_smoke_197.py
LOG=${WORK_DIR}/launch.log

mkdir -p "${WORK_DIR}"
if find "${WORK_DIR}" -mindepth 1 -maxdepth 1 ! -name launch.log | grep -q .; then
    echo "Refusing a fresh smoke launch into non-empty ${WORK_DIR}" >&2
    exit 2
fi

cd "${REPO}"
grep -q 'init_pair_structural_weights' \
    projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr.py

set +u
source /data/users/litianhao/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u

export CUDA_VISIBLE_DEVICES=4,5
export PORT=29950
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] fresh 0729_04 old-DN 4-iter smoke" >> "${LOG}"
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
python projects/multispec_pair_rotated_rtdetr/tools/verify_tristate_decoder_smoke_init.py \
    "${CHECKPOINT}" >> "${LOG}" 2>&1
