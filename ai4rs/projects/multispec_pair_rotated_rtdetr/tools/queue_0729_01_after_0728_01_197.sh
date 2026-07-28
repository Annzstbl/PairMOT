#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/litianhao/PairMOT/ai4rs
TOOLS=${REPO}/projects/multispec_pair_rotated_rtdetr/tools
WAIT_PATTERN='tools/train.py.*0728_01_paper_base_liquid_encoder_p5temporal_dualevidence_decoder0708_03'
SMOKE_LOG=/data4/litianhao/PairMmot/workdir_197/smoke_0729_01_base_liquid_encoder_dualevidence_decoder0708_03_easyhardpositive_initfix71c69b4_4iter/launch.log

gpu_busy() {
    nvidia-smi -i "$1" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'
}

echo "[$(date '+%F %T')] waiting for 0728_01 to release GPU 4,5"
while pgrep -f -- "${WAIT_PATTERN}" >/dev/null; do
    sleep 60
done
while gpu_busy 4 || gpu_busy 5; do
    echo "[$(date '+%F %T')] training exited but GPU 4,5 still busy"
    sleep 60
done

echo "[$(date '+%F %T')] launching fixed-init 0729_01 smoke"
bash "${TOOLS}/launch_smoke_0729_01_encoder_dualevidence_decoder0708_03_easyhardpositive_initfix_197.sh"
grep -q 'TRISTATE_SMOKE_INIT_OK' "${SMOKE_LOG}"

echo "[$(date '+%F %T')] smoke passed; launching formal 0729_01"
bash "${TOOLS}/launch_0729_01_encoder_dualevidence_decoder0708_03_easyhardpositive_197.sh"
