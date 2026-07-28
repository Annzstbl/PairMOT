#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=/data/users/litianhao/data/HSMOT
CACHE_ROOT=/dev/shm/litianhao/pairmot_hsmot_cache
LOCK=/dev/shm/litianhao/.pairmot_hsmot_cache.lock
RESERVE_BYTES=$((32 * 1024 * 1024 * 1024))
VERSION=1

mkdir -p "$(dirname "${CACHE_ROOT}")"
chmod 700 "$(dirname "${CACHE_ROOT}")"
exec 9>"${LOCK}"
flock 9

measure_split() {
    local root=$1
    local count bytes
    count=$(find -L "${root}" -type f | wc -l)
    bytes=$(find -L "${root}" -type f -printf '%s\n' \
        | awk '{total += $1} END {printf "%.0f", total}')
    printf '%s %s\n' "${count}" "${bytes}"
}

read -r train_count train_bytes < <(
    measure_split "${SOURCE_ROOT}/train/npy2jpg")
read -r test_count test_bytes < <(
    measure_split "${SOURCE_ROOT}/test/npy2jpg")
fingerprint=$(printf \
    'version=%s\nsource=%s\ntrain_count=%s\ntrain_bytes=%s\ntest_count=%s\ntest_bytes=%s\n' \
    "${VERSION}" "${SOURCE_ROOT}" "${train_count}" "${train_bytes}" \
    "${test_count}" "${test_bytes}")

cache_valid() {
    [[ -f "${CACHE_ROOT}/.pairmot_cache_ready" ]] || return 1
    [[ -L "${CACHE_ROOT}/train/mot" && -L "${CACHE_ROOT}/test/mot" ]] \
        || return 1
    [[ -d "${CACHE_ROOT}/train/npy2jpg" \
        && -d "${CACHE_ROOT}/test/npy2jpg" ]] || return 1
    [[ "$(cat "${CACHE_ROOT}/.pairmot_cache_ready")" == "${fingerprint}" ]] \
        || return 1
    local cached_train_count cached_train_bytes
    local cached_test_count cached_test_bytes
    read -r cached_train_count cached_train_bytes < <(
        measure_split "${CACHE_ROOT}/train/npy2jpg")
    read -r cached_test_count cached_test_bytes < <(
        measure_split "${CACHE_ROOT}/test/npy2jpg")
    [[ "${cached_train_count}" == "${train_count}" \
        && "${cached_train_bytes}" == "${train_bytes}" \
        && "${cached_test_count}" == "${test_count}" \
        && "${cached_test_bytes}" == "${test_bytes}" ]]
}

if cache_valid; then
    echo "Validated existing HSMOT tmpfs cache: ${CACHE_ROOT}" >&2
    printf '%s\n' "${CACHE_ROOT}"
    exit 0
fi

required_bytes=$((train_bytes + test_bytes))
available_bytes=$(df --output=avail -B1 /dev/shm | tail -1 | tr -d ' ')
if (( available_bytes < required_bytes + RESERVE_BYTES )); then
    echo "Insufficient /dev/shm space: available=${available_bytes}, required=${required_bytes}, reserve=${RESERVE_BYTES}" >&2
    exit 10
fi

STAGING="${CACHE_ROOT}.staging.$$"
OLD="${CACHE_ROOT}.old.$$"
cleanup() {
    rm -rf -- "${STAGING}"
}
trap cleanup EXIT

mkdir -p "${STAGING}/train/npy2jpg" "${STAGING}/test/npy2jpg"
chmod 700 "${STAGING}"
ln -s "${SOURCE_ROOT}/train/mot" "${STAGING}/train/mot"
ln -s "${SOURCE_ROOT}/test/mot" "${STAGING}/test/mot"
rsync -aL --delete "${SOURCE_ROOT}/train/npy2jpg/" \
    "${STAGING}/train/npy2jpg/"
rsync -aL --delete "${SOURCE_ROOT}/test/npy2jpg/" \
    "${STAGING}/test/npy2jpg/"

read -r copied_train_count copied_train_bytes < <(
    measure_split "${STAGING}/train/npy2jpg")
read -r copied_test_count copied_test_bytes < <(
    measure_split "${STAGING}/test/npy2jpg")
[[ "${copied_train_count}" == "${train_count}" \
    && "${copied_train_bytes}" == "${train_bytes}" \
    && "${copied_test_count}" == "${test_count}" \
    && "${copied_test_bytes}" == "${test_bytes}" ]]
printf '%s' "${fingerprint}" > "${STAGING}/.pairmot_cache_ready"

if [[ -e "${CACHE_ROOT}" ]]; then
    mv "${CACHE_ROOT}" "${OLD}"
fi
mv "${STAGING}" "${CACHE_ROOT}"
trap - EXIT
if [[ -e "${OLD}" ]]; then
    rm -rf -- "${OLD}"
fi

cache_valid
echo "Created and validated HSMOT tmpfs cache: ${CACHE_ROOT}" >&2
printf '%s\n' "${CACHE_ROOT}"
