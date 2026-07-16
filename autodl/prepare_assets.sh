#!/usr/bin/env bash
set -Eeuo pipefail

: "${AI4RS_ROOT:?}"
: "${HSMOT_ROOT:?}"
: "${PRETRAIN_ROOT:?}"
: "${GMC_ROOT:?}"

mkdir -p "$PRETRAIN_ROOT" "$GMC_ROOT"
ASSET_ROOT="${ASSET_ROOT:-$(dirname -- "$PRETRAIN_ROOT")}"
GMC_ARCHIVE="${GMC_ARCHIVE:-$ASSET_ROOT/gmc_cache_hsmot_gap1.tar.gz}"
SOURCE_NAME=rtdetr_r18vd_dec3_6x_coco_from_paddle.pth
SOURCE_PATH="$PRETRAIN_ROOT/$SOURCE_NAME"
SOURCE_URL="https://github.com/lyuwenyu/storage/releases/download/v0.1/$SOURCE_NAME"
SOURCE_SHA256=3ba8b5c909c9a1c4f21e96d0a7251ab1a485093955ca327d0061fef8d33c66f0
ADAPTED="$PRETRAIN_ROOT/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth"

source_is_valid() {
  [[ -f "$SOURCE_PATH" ]] && \
    echo "$SOURCE_SHA256  $SOURCE_PATH" | sha256sum --check --status
}

if [[ ! -f "$ADAPTED" ]]; then
  if ! source_is_valid; then
    echo "Downloading official RT-DETR R18 COCO checkpoint..."
    curl -fL --retry 5 --retry-delay 3 -o "$SOURCE_PATH.part" "$SOURCE_URL"
    echo "$SOURCE_SHA256  $SOURCE_PATH.part" | sha256sum --check --status || {
      echo "Downloaded checkpoint checksum mismatch" >&2
      exit 1
    }
    mv "$SOURCE_PATH.part" "$SOURCE_PATH"
  fi
  (cd "$AI4RS_ROOT" && python \
    projects/multispec_pair_rotated_rtdetr/tools/prepare_coco_pair_family_pretrain.py \
    --backbones r18 --pretrain-root "$PRETRAIN_ROOT")
else
  echo "Adapted pretrain already exists: $ADAPTED"
fi

if [[ "${BUILD_GMC:-1}" == 1 ]]; then
  if [[ -f "$GMC_ARCHIVE" ]]; then
    archive_dir=$(dirname -- "$GMC_ARCHIVE")
    archive_name=$(basename -- "$GMC_ARCHIVE")
    if [[ -f "$GMC_ARCHIVE.sha256" ]]; then
      (cd "$archive_dir" && sha256sum --check "$archive_name.sha256")
    fi
    if [[ ! -d "$GMC_ROOT/hsmot_train_gap1" || \
          ! -d "$GMC_ROOT/hsmot_test_gap1" ]]; then
      echo "Restoring GMC cache from shared storage: $GMC_ARCHIVE"
      tar -xzf "$GMC_ARCHIVE" -C "$GMC_ROOT"
    fi
  fi

  for split in train test; do
    out="$GMC_ROOT/hsmot_${split}_gap1"
    echo "Preparing real sparse-LK/RANSAC GMC for $split..."
    (cd "$AI4RS_ROOT" && python \
      projects/multispec_pair_rotated_rtdetr/tools/build_hsmot_gmc_cache.py \
      --data-root "$HSMOT_ROOT/$split" --ann-subdir mot \
      --img-subdir npy2jpg --img-format 3jpg --out-dir "$out" --gaps 1)
    count=$(find "$out" -type f -name '*.json' | wc -l)
    [[ "$count" -gt 0 ]] || { echo "No GMC entries generated for $split" >&2; exit 1; }
    echo "GMC $split entries: $count"
    python "$(dirname -- "${BASH_SOURCE[0]}")/validate_gmc.py" \
      --data-root "$HSMOT_ROOT/$split" --cache-root "$out"
  done

  if [[ ! -f "$GMC_ARCHIVE" ]]; then
    mkdir -p "$(dirname -- "$GMC_ARCHIVE")"
    archive_tmp="$GMC_ARCHIVE.part.$$"
    echo "Archiving validated GMC cache to shared storage..."
    tar -C "$GMC_ROOT" -czf "$archive_tmp" \
      hsmot_train_gap1 hsmot_test_gap1
    mv "$archive_tmp" "$GMC_ARCHIVE"
    archive_dir=$(dirname -- "$GMC_ARCHIVE")
    archive_name=$(basename -- "$GMC_ARCHIVE")
    (cd "$archive_dir" && sha256sum "$archive_name" > "$archive_name.sha256")
    echo "Persistent GMC archive: $GMC_ARCHIVE"
  fi
else
  echo "BUILD_GMC=0: GMC preparation skipped; formal training config will not run yet."
fi
