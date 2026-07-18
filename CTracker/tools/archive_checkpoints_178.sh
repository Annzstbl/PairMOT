#!/usr/bin/env bash
set -u

local_dir=$1
archive_dir=$2
done_file=$3

# This worker is deliberately independent from training. If NFS /data4 is
# unavailable, rsync may block here while training continues on local NVMe.
mkdir -p "${archive_dir}"

archive_file() {
  local source=$1
  [[ -f "${source}" ]] || return 0
  rsync -a --remove-source-files --partial-dir=.rsync-partial \
    "${source}" "${archive_dir}/"
}

while true; do
  shopt -s nullglob
  epoch_checkpoints=("${local_dir}"/checkpoint_epoch_*.pt)
  shopt -u nullglob
  for checkpoint in "${epoch_checkpoints[@]}"; do
    archive_file "${checkpoint}"
  done

  if [[ -e "${done_file}" ]]; then
    archive_file "${local_dir}/checkpoint_latest.pt"
    archive_file "${local_dir}/model_final.pt"
    archive_file "${local_dir}/train.log"

    shopt -s nullglob
    remaining=("${local_dir}"/checkpoint_epoch_*.pt)
    shopt -u nullglob
    if (( ${#remaining[@]} == 0 )); then
      exit 0
    fi
  fi
  sleep 30
done
