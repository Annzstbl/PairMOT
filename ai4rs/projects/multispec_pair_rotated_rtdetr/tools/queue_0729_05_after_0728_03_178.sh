#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/users/litianhao01/PairMOT/ai4rs
TOOLS=${REPO}/projects/multispec_pair_rotated_rtdetr/tools
WAIT_PATTERN='tools/train.py.*dualevidence_pairdn_easyhardpositive.*178.py'
SMOKE_LOG=/data4/litianhao/PairMmot/workdir_178/smoke_0729_05_liquid_independent_diffproduct_pairdn_easyhardpositive_4iter/launch.log

gpu_busy() {
    nvidia-smi -i 0 --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'
}

echo "[$(date '+%F %T')] waiting for 0728_03 to release GPU 0"
while pgrep -f -- "${WAIT_PATTERN}" >/dev/null; do
    sleep 15
done
while gpu_busy; do
    echo "[$(date '+%F %T')] 0728_03 exited but GPU 0 still busy"
    sleep 15
done

echo "[$(date '+%F %T')] launching 0729_05 recovery smoke"
bash "${TOOLS}/launch_smoke_0729_05_easyhardpositive_control_178.sh"
grep -q 'PAIRDNEASYPOS_SMOKE_OK' "${SMOKE_LOG}"

echo "[$(date '+%F %T')] smoke passed; launching formal 0729_05"
bash "${TOOLS}/launch_0729_05_easyhardpositive_control_178.sh"
