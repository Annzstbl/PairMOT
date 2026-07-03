#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/users/litianhao01/PairMmot
AI4RS=${ROOT}/ai4rs
WORK_ROOT=${ROOT}/workdir
GMC_ROOT=${WORK_ROOT}/aux/gmc_cache
GPU_LIST=${GPU_LIST:-2,3}
NUM_GPUS=${NUM_GPUS:-2}
PORT_BASE=${PORT_BASE:-29820}

set +u
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u
cd "${AI4RS}"

mkdir -p "${GMC_ROOT}"

build_cache_if_needed() {
  local split=$1
  local data_root=$2
  local ann_file=$3
  local out_dir=$4
  if find "${out_dir}" -name '*.json' -print -quit 2>/dev/null | grep -q .; then
    echo "[gmc] reuse ${split}: ${out_dir}"
    return
  fi
  echo "[gmc] build ${split}: ${out_dir}"
  python projects/multispec_pair_rotated_rtdetr/tools/build_hsmot_gmc_cache.py \
    --data-root "${data_root}" \
    --ann-file "${ann_file}" \
    --ann-subdir mot \
    --img-subdir npy2jpg \
    --img-format 3jpg \
    --gaps 1 \
    --out-dir "${out_dir}"
}

build_cache_if_needed \
  train \
  "${ROOT}/data/hsmot/train" \
  "../train_half.txt" \
  "${GMC_ROOT}/hsmot_train_gap1"

build_cache_if_needed \
  test \
  "${ROOT}/data/hsmot/test" \
  "" \
  "${GMC_ROOT}/hsmot_test_gap1"

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
  projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2.py \
  "${WORK_ROOT}/0702_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2"

run_exp 1 \
  projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_pairdn.py \
  "${WORK_ROOT}/0702_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_pairdn"
