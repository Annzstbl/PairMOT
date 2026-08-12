#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${PAIRMOT_REPO:-/data/users/litianhao/PairMOT_0812_01_wsd_resume197/ai4rs}
CONFIG=projects/multispec_pair_rotated_rtdetr/configs/smoke/o2_pair_rtdetr_r18vd_0812_01_product_tangent_wsd_resume_e36_4iter_smoke_197.py
WORK_DIR=/data4/litianhao/PairMmot/workdir_197/smoke_0812_01_product_tangent_wsd_resume_e36_4iter
RESUME=/data4/litianhao/PairMmot/workdir_252/0810_09_final_product_tangent_wsd4_56_cos12_72e_2xb4_fresh/epoch_36.pth

set +u
source /data/users/litianhao/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
test "${CONDA_DEFAULT_ENV}" = py310
test "${CONDA_PREFIX}" = /data/users/litianhao/anaconda3/envs/py310
test ! -e "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
trap 'status=$?; echo "[$(date "+%F %T")] resume smoke failed: status=${status} command=${BASH_COMMAND}" >> "${WORK_DIR}/launch.log"; exit "${status}"' ERR

cd "${REPO}"
test -f "${RESUME}"
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG TORCH_DISTRIBUTED_DEBUG
"${CONDA_PREFIX}/bin/torchrun" --nproc_per_node=2 --master_port=29920 \
    tools/train.py "${CONFIG}" --launcher pytorch --work-dir "${WORK_DIR}" \
    --resume "${RESUME}" >> "${WORK_DIR}/launch.log" 2>&1
grep -q 'resumed epoch: 36, iter: 37368' "${WORK_DIR}/launch.log"
grep -q 'Iter(train).*37372/37372' "${WORK_DIR}/launch.log"
test -s "${WORK_DIR}/iter_37372.pth"
"${CONDA_PREFIX}/bin/python" projects/multispec_pair_rotated_rtdetr/tools/check_checkpoint_all_finite.py "${WORK_DIR}/iter_37372.pth" >> "${WORK_DIR}/launch.log" 2>&1
