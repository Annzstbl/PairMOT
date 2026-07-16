#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# AutoDL's non-interactive SSH shell may omit the image's base Python from
# PATH even though it is available under /root/miniconda3. Use that existing
# environment directly; no conda environment is created or activated.
if ! command -v python >/dev/null 2>&1 && \
   [[ -x /root/miniconda3/bin/python ]]; then
  export PATH="/root/miniconda3/bin:$PATH"
fi
command -v python >/dev/null 2>&1 || {
  echo "Python executable not found in the AutoDL image" >&2
  exit 1
}
if [[ -z "${FS_ROOT:-}" ]]; then
  for candidate in /root/autodl-fs /root/autoldl-fs /autodl-fs; do
    if [[ -d "$candidate" ]]; then
      FS_ROOT="$candidate"
      break
    fi
  done
  FS_ROOT="${FS_ROOT:-/root/autodl-fs}"
fi
export FS_ROOT
export PAIRMOT_REPO_URL="${PAIRMOT_REPO_URL:-https://github.com/Annzstbl/PairMOT.git}"
export PAIRMOT_REF="${PAIRMOT_REF:-main}"
export PAIRMOT_ROOT="${PAIRMOT_ROOT:-/root/PairMOT}"
export AI4RS_ROOT="$PAIRMOT_ROOT/ai4rs"
export HSMOT_ARCHIVE="${HSMOT_ARCHIVE:-$FS_ROOT/hsmot.tar.gz}"
export HSMOT_ROOT="${HSMOT_ROOT:-/root/autodl-tmp/data/hsmot}"
export ASSET_ROOT="${ASSET_ROOT:-$FS_ROOT/PairMOT_assets}"
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-$ASSET_ROOT/pretrained_weights}"
export GMC_ROOT="${GMC_ROOT:-/root/autodl-tmp/PairMOT_assets/gmc_cache}"
export SMOKE_WORK_DIR="${SMOKE_WORK_DIR:-/root/autodl-tmp/work_dirs/smoke_paper_base}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/PairMOT_setup_logs}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/bootstrap_$(date +%Y%m%d_%H%M%S).log") 2>&1

echo "[1/7] Cloning PairMOT ($PAIRMOT_REF)"
if [[ ! -d "$PAIRMOT_ROOT/.git" ]]; then
  git clone "$PAIRMOT_REPO_URL" "$PAIRMOT_ROOT"
  git -C "$PAIRMOT_ROOT" checkout "$PAIRMOT_REF"
else
  echo "Existing clone retained: $PAIRMOT_ROOT ($(git -C "$PAIRMOT_ROOT" rev-parse --short HEAD))"
fi
[[ -f "$AI4RS_ROOT/setup.py" ]] || { echo "Invalid repository: $AI4RS_ROOT" >&2; exit 1; }

echo "[2/7] Normalizing HSMOT archive layout"
python "$SCRIPT_DIR/prepare_hsmot.py" --archive "$HSMOT_ARCHIVE" --target "$HSMOT_ROOT"

echo "[3/7] Installing MM stack and ai4rs (image torch is retained)"
if [[ "${SKIP_INSTALL:-0}" == 1 ]]; then
  echo "SKIP_INSTALL=1: retained image environment is used."
else
  bash "$SCRIPT_DIR/install_environment.sh"
fi

echo "[4/7] Preparing official pretrain and real GMC cache"
bash "$SCRIPT_DIR/prepare_assets.sh"

echo "[5/7] Installing AutoDL runtime configuration"
CONFIG_DIR="$AI4RS_ROOT/projects/multispec_pair_rotated_rtdetr/configs"
RUNTIME_CONFIG="$CONFIG_DIR/autodl_paper_base_r18_coco_full_1200x900_bf16.py"
FORMAL_WORK_DIR="${WORK_DIR:-/root/autodl-tmp/work_dirs/paper_base_r18_coco_full_1200x900}"
python "$SCRIPT_DIR/render_runtime_config.py" \
  --template "$SCRIPT_DIR/paper_base_autodl.py.template" \
  --output "$RUNTIME_CONFIG" --hsmot-root "$HSMOT_ROOT" \
  --pretrain-root "$PRETRAIN_ROOT" --gmc-root "$GMC_ROOT" \
  --work-dir "$FORMAL_WORK_DIR"
for path in \
  "$PRETRAIN_ROOT/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/pair_coco_adapted_pretrain.pth" \
  "$GMC_ROOT/hsmot_train_gap1" "$GMC_ROOT/hsmot_test_gap1"; do
  [[ -e "$path" ]] || { echo "Required formal-training asset missing: $path" >&2; exit 1; }
done

echo "[6/7] Writing persistent shell environment"
ENV_FILE="$PAIRMOT_ROOT/autodl_runtime.env"
cat > "$ENV_FILE" <<EOF
export PAIRMOT_ROOT='$PAIRMOT_ROOT'
export AI4RS_ROOT='$AI4RS_ROOT'
export FS_ROOT='$FS_ROOT'
export HSMOT_ROOT='$HSMOT_ROOT'
export PRETRAIN_ROOT='$PRETRAIN_ROOT'
export GMC_ROOT='$GMC_ROOT'
export PYTHONPATH='$AI4RS_ROOT':\${PYTHONPATH:-}
export AUTODL_PAIRMOT_CONFIG='$RUNTIME_CONFIG'
EOF
SOURCE_LINE="source '$ENV_FILE'"
grep -Fqx "$SOURCE_LINE" "$HOME/.bashrc" || printf '\n%s\n' "$SOURCE_LINE" >> "$HOME/.bashrc"
source "$ENV_FILE"

echo "[7/7] Running baseline train/inference smoke test"
(cd "$AI4RS_ROOT" && python "$SCRIPT_DIR/smoke_baseline.py" \
  --config "$RUNTIME_CONFIG" --work-dir "$SMOKE_WORK_DIR")

echo "AutoDL initialization completed."
echo "Formal single-GPU training command:"
echo "  cd '$AI4RS_ROOT' && python tools/train.py '$RUNTIME_CONFIG'"
