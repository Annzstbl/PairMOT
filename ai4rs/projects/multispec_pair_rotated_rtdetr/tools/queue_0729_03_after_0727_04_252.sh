#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/litianhao01/PairMmot/ai4rs
TOOLS=${REPO}/projects/multispec_pair_rotated_rtdetr/tools
WAIT_PATTERN='tools/train.py.*0727_04_paper_base_liquid_encoder_p5temporal_detailenergy'
SMOKE_LOG=/data4/litianhao/PairMmot/workdir_252/smoke_0729_03_dualevidence_decoder_dualoutputresidual_paircoherent_4iter/launch.log

gpu_busy() {
    nvidia-smi -i "$1" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'
}

echo "[$(date '+%F %T')] waiting for 0727_04 to release GPU 0,1"
while pgrep -f -- "${WAIT_PATTERN}" >/dev/null; do
    sleep 60
done
while gpu_busy 0 || gpu_busy 1; do
    echo "[$(date '+%F %T')] training exited but GPU 0,1 still busy"
    sleep 60
done

echo "[$(date '+%F %T')] launching strict residual 0729_03 smoke"
bash "${TOOLS}/launch_smoke_0729_03_dualoutputresidual_paircoherent_252.sh"
grep -q 'DUAL_OUTPUT_ADAPTER_SMOKE_OK' "${SMOKE_LOG}"

echo "[$(date '+%F %T')] smoke passed; launching formal 0729_03"
bash "${TOOLS}/launch_0729_03_dualoutputresidual_paircoherent_252.sh"
