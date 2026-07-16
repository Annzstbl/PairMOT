#!/usr/bin/env bash
set -Eeuo pipefail

: "${AI4RS_ROOT:?AI4RS_ROOT must point to the cloned PairMOT/ai4rs directory}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_INDEX_URL="${PACKAGE_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}}"
if [[ -d /usr/local/cuda/bin ]]; then
  export PATH="/usr/local/cuda/bin:$PATH"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

python - <<'PY'
from importlib.metadata import version
from packaging.version import Version
import torch

if Version(torch.__version__.split('+')[0]) < Version('2.2.0'):
    raise SystemExit(f'AutoDL image torch must be >=2.2.0, got {torch.__version__}')
print(f'torch={torch.__version__}, cuda={torch.version.cuda}, '
      f'cuda_available={torch.cuda.is_available()}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Select a GPU image before initialization.')
PY

# OpenMIM's optional OpenDataLab dependency pins a Python-3.12-incompatible
# setuptools release. The MM installation path only needs the CLI dependencies.
python -m pip install -i "$PACKAGE_INDEX_URL" -U \
  "pip<26" "numpy<2" "setuptools>=69,<81" "requests>=2.31" "urllib3>=2"
python -m pip uninstall -y opendatalab openxlab
python -m pip install -i "$PACKAGE_INDEX_URL" --no-deps openmim
python -m pip install -i "$PACKAGE_INDEX_URL" \
  click colorama model-index rich tabulate
# mmcv-lite and mmcv share the same Python namespace; retaining both can make
# CUDA ops appear installed while importing the lite package.
python -m pip uninstall -y mmcv-lite || true
mim install "mmengine>=0.10.0,<1.0.0" -i "$PACKAGE_INDEX_URL"
if ! python -c "import mmcv; assert mmcv.__version__ == '2.2.0'" \
    >/dev/null 2>&1; then
  TORCH_TAG=$(python -c \
    "import torch; print(torch.__version__.split('+')[0])")
  CUDA_TAG=$(python -c \
    "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
  PYTHON_TAG=$(python -c \
    "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  CUDA_ARCH=$(python -c \
    "import torch; a=torch.cuda.get_device_capability(); print(f'{a[0]}.{a[1]}')")
  WHEEL_ROOT="${MMCV_WHEEL_ROOT:-${ASSET_ROOT:-/root/autodl-fs/PairMOT_assets}/wheels}"
  WHEEL_DIR="$WHEEL_ROOT/mmcv-2.2.0_torch${TORCH_TAG}_${CUDA_TAG}_${PYTHON_TAG}_sm${CUDA_ARCH/.}"
  mkdir -p "$WHEEL_DIR"
  shopt -s nullglob
  wheels=("$WHEEL_DIR"/mmcv-2.2.0-*.whl)
  shopt -u nullglob
  if (( ${#wheels[@]} == 0 )); then
    echo "No compatible MMCV wheel found; compiling CUDA ops for sm_$CUDA_ARCH"
    python -m pip install -i "$PACKAGE_INDEX_URL" ninja
    export MMCV_WITH_OPS=1
    export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
    export MAX_JOBS="${MAX_JOBS:-8}"
    python -m pip wheel -v --no-build-isolation --no-deps \
      -i "$PACKAGE_INDEX_URL" --wheel-dir "$WHEEL_DIR" "mmcv==2.2.0"
    wheels=("$WHEEL_DIR"/mmcv-2.2.0-*.whl)
  fi
  python -m pip install "${wheels[0]}"
fi
mim install "mmdet>=3.1.0,<3.4.0" -i "$PACKAGE_INDEX_URL"
mim install "mmsegmentation==1.2.2" -i "$PACKAGE_INDEX_URL"
python "$SCRIPT_DIR/patch_mm_compat.py"

# Install runtime requirements explicitly, excluding torch supplied by the image.
python -m pip install -i "$PACKAGE_INDEX_URL" \
  matplotlib pycocotools six terminaltables scipy \
  "numpy<2" "opencv-python<4.12" ftfy regex
python -m pip install -v -e "$AI4RS_ROOT" --no-deps

python - <<'PY'
from importlib.metadata import version
import torch
import mmcv
import mmengine
import mmdet
import mmseg
from mmcv.ops import MultiScaleDeformableAttention

assert mmcv.__version__ == '2.2.0', mmcv.__version__
assert torch.cuda.is_available()
print('Environment verification passed:')
for name in ('torch', 'mmcv', 'mmengine', 'mmdet', 'mmsegmentation', 'mmrotate'):
    print(f'  {name}={version(name)}')
print(f'  mmcv CUDA ops={MultiScaleDeformableAttention.__module__}')
PY
