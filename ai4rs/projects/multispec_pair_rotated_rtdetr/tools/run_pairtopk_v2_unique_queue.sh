#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/users/litianhao01/PairMmot
AI4RS=${ROOT}/ai4rs
WORK_ROOT=${ROOT}/workdir
GMC_ROOT=${WORK_ROOT}/aux/gmc_cache
GPU_LIST=${GPU_LIST:-2,3}
NUM_GPUS=${NUM_GPUS:-2}
PORT_BASE=${PORT_BASE:-29840}

set +u
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
cd "${AI4RS}"

mkdir -p "${GMC_ROOT}"

run_exp() {
  local idx=$1
  local config=$2
  local work_dir=$3
  local port=$((PORT_BASE + idx))
  mkdir -p "${work_dir}"
  echo "[train] start ${config}"
  echo "[train] work_dir=${work_dir} gpu=${GPU_LIST} port=${port}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" PORT="${port}" bash tools/dist_train.sh \
    "${config}" "${NUM_GPUS}" --work-dir "${work_dir}"
  echo "[train] done ${config}"
}

run_exp 0 \
  projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique.py \
  "${WORK_ROOT}/0702_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique"

run_exp 1 \
  projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn.py \
  "${WORK_ROOT}/0702_baseline_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn"
