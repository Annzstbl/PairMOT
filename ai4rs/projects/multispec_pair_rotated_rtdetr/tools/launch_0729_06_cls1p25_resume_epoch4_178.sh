#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
SOURCE=/data4/litianhao/PairMmot/workdir_197/0729_06_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_dualoutputresidual_cls1p25_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_orderedpairs_fresh/epoch_4.pth
WORK_DIR=/data4/litianhao/PairMmot/workdir_178/0729_06_recovery_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_dualoutputresidual_cls1p25_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_from_epoch4
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0729_06_cls1p25_resume_178.py
LOG=${WORK_DIR}/launch.log

test -f "${SOURCE}"
mkdir -p "${WORK_DIR}"
if find "${WORK_DIR}" -mindepth 1 -maxdepth 1 ! -name launch.log | grep -q .; then
    echo "Refusing recovery launch into non-empty ${WORK_DIR}" >&2
    exit 2
fi
if ps -eo comm=,args= | awk \
        '$1 ~ /^python/ && $0 ~ /tools\/(train|test)[.]py/ { found = 1 }
         END { exit !found }'; then
    echo "Refusing recovery while PairMOT train/test is active on 178" >&2
    exit 3
fi

set +u
source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
cd "${REPO}"

test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
test -d /data1/users/litianhao01/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUBLAS_WORKSPACE_CONFIG
unset TORCH_DISTRIBUTED_DEBUG

echo "[$(date '+%F %T')] recover 0729_06 cls1.25 from 197 epoch4 as 1x8" \
    >> "${LOG}"
python tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" \
    --resume "${SOURCE}" >> "${LOG}" 2>&1
