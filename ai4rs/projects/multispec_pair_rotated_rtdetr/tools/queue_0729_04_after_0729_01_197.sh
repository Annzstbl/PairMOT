#!/usr/bin/env bash
set -euo pipefail

REPO=/data/users/litianhao/PairMOT/ai4rs
TOOLS=${REPO}/projects/multispec_pair_rotated_rtdetr/tools
WAIT_PATTERN='tools/train.py.*decoder0708_03_pairdn_easyhardpositive.*197.py'
SMOKE_LOG=/data4/litianhao/PairMmot/workdir_197/smoke_0729_04_base_liquid_encoder_dualevidence_decoder0708_03_initfix_pairdn_paircoherent_4iter/launch.log

gpu_busy() {
    nvidia-smi -i "$1" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'
}

echo "[$(date '+%F %T')] waiting for 0729_01 to release GPU 4,5"
while pgrep -f -- "${WAIT_PATTERN}" >/dev/null; do
    sleep 15
done
while gpu_busy 4 || gpu_busy 5; do
    echo "[$(date '+%F %T')] 0729_01 exited but GPU 4,5 still busy"
    sleep 15
done

echo "[$(date '+%F %T')] launching strict old-DN fixed-init 0729_04 smoke"
bash "${TOOLS}/launch_smoke_0729_04_tristate_initfix_paircoherent_197.sh"
grep -q 'TRISTATE_SMOKE_INIT_OK' "${SMOKE_LOG}"

echo "[$(date '+%F %T')] smoke passed; launching formal 0729_04"
bash "${TOOLS}/launch_0729_04_tristate_initfix_paircoherent_197.sh"
