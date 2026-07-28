#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178.py
SOURCE_DIR=/data4/litianhao/PairMmot/workdir_178/0727_01_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh
SOURCE_CONFIG=o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178.py
SOURCE_CKPT=${SOURCE_DIR}/epoch_8.pth
ROOT=/data4/litianhao/PairMmot/workdir_178/diag_0727_01_epoch8_encoder_branches
TOOL=projects/multispec_pair_rotated_rtdetr/tools/make_encoder_branch_ablation_checkpoint.py

if pgrep -af "tools/train.py.*${SOURCE_CONFIG}" >/dev/null; then
    echo "Refusing diagnostics while 0727_01 training is active" >&2
    exit 2
fi
test -f "${SOURCE_CKPT}"
eval_count=$(find "${SOURCE_DIR}/val_track_eval" \
    -type d -name eval 2>/dev/null | wc -l)
if (( eval_count < 2 )); then
    echo "Expected epoch 4/8 TrackEval before diagnostics, got ${eval_count}" >&2
    exit 3
fi
[[ ! -e "${ROOT}" ]] || {
    echo "Refusing to overwrite existing ${ROOT}" >&2
    exit 4
}
mkdir -p "${ROOT}/checkpoints"

set +u
source /data1/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
cd "${REPO}"
export PAIRMOT_HSMOT_ROOT=/data1/users/litianhao01/data/hsmot
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for mode in no_common no_detail no_p4_common no_post; do
    checkpoint="${ROOT}/checkpoints/${mode}.pth"
    output_dir="${ROOT}/${mode}"
    python "${TOOL}" "${SOURCE_CKPT}" "${checkpoint}" \
        --mode "${mode}" > "${ROOT}/${mode}_checkpoint.log"
    mkdir -p "${output_dir}"
    python tools/test.py "${CONFIG}" "${checkpoint}" \
        --work-dir "${output_dir}" \
        --cfg-options \
        "test_evaluator.metrics.track_eval_out_dir=${output_dir}/val_track_eval" \
        "test_evaluator.metrics.val_det_out_dir=${output_dir}/val_det" \
        test_evaluator.metrics.async_track_eval=False \
        > "${output_dir}/test.log" 2>&1
    test -f "${output_dir}/val_track_eval/val_track_0001/metrics.json"
    test -f "${output_dir}/val_det/val_0001/metrics.json"
    test -d "${output_dir}/val_track_eval/val_track_0001/trackers"
done

echo "[$(date '+%F %T')] completed analysis-only epoch8 branch diagnostics" \
    > "${ROOT}/COMPLETE"
