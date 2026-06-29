#!/usr/bin/env bash
# Move repo-local cache/ and outputs/ to shared team storage, symlink back, and
# write CSD_CACHE_ROOT / CSD_OUTPUTS_ROOT into synthesis/.env.
#
# Logs stay under ./logs/ (not migrated).
#
# Usage (from repo root):
#   ./migrate_to_shared.sh
#
# Override destinations:
#   CSD_CACHE_ROOT=/path/to/cache CSD_OUTPUTS_ROOT=/path/to/outputs ./migrate_to_shared.sh
#
# Requires: rsync, write access to the shared paths and synthesis/.env

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_CACHE="${CSD_CACHE_ROOT:-/share/metadecode/cache}"
SHARED_OUTPUTS="${CSD_OUTPUTS_ROOT:-/share/metadecode/outputs}"
ENV_FILE="${ROOT}/synthesis/.env"

echo "=== migrate_to_shared ==="
echo "repo:           ${ROOT}"
echo "shared cache:   ${SHARED_CACHE}"
echo "shared outputs: ${SHARED_OUTPUTS}"
echo "env file:       ${ENV_FILE}"
echo "logs:           ${ROOT}/logs/ (unchanged, stays local)"
echo

mkdir -p "${SHARED_CACHE}" "${SHARED_OUTPUTS}"

migrate_tree() {
  local name="$1"
  local src="${ROOT}/${name}"
  local dest="$2"

  if [[ -L "${src}" ]]; then
    local target
    target="$(readlink -f "${src}" 2>/dev/null || readlink "${src}")"
    echo "[${name}] already symlink: ${src} -> ${target}"
    if [[ "${target}" != "$(readlink -f "${dest}" 2>/dev/null || echo "${dest}")" ]]; then
      echo "[${name}] WARNING: symlink target differs from ${dest}" >&2
    fi
    return 0
  fi

  if [[ ! -e "${src}" ]]; then
    echo "[${name}] missing ${src}; creating symlink -> ${dest}"
    ln -sfn "${dest}" "${src}"
    return 0
  fi

  if [[ -d "${src}" ]] && [[ -n "$(find "${src}" -mindepth 1 -maxdepth 1 2>/dev/null | head -1)" ]]; then
    echo "[${name}] rsync ${src}/ -> ${dest}/"
    mkdir -p "${dest}"
    rsync -a "${src}/" "${dest}/"
    echo "[${name}] removing ${src}"
    rm -rf "${src}"
  elif [[ -d "${src}" ]]; then
    echo "[${name}] empty directory ${src}"
    rmdir "${src}" 2>/dev/null || rm -rf "${src}"
  else
    echo "[${name}] ERROR: ${src} exists but is not a directory" >&2
    exit 1
  fi

  ln -sfn "${dest}" "${src}"
  echo "[${name}] symlink ${src} -> ${dest}"
}

set_env_var() {
  local key="$1"
  local value="$2"
  touch "${ENV_FILE}"
  if grep -qE "^(export )?${key}=" "${ENV_FILE}"; then
    sed -i -E "s|^(export )?${key}=.*|\1${key}=\"${value}\"|" "${ENV_FILE}"
  else
    printf '%s=\"%s\"\n' "${key}" "${value}" >> "${ENV_FILE}"
  fi
}

migrate_tree cache "${SHARED_CACHE}"
migrate_tree outputs "${SHARED_OUTPUTS}"

set_env_var CSD_CACHE_ROOT "${SHARED_CACHE}"
set_env_var CSD_OUTPUTS_ROOT "${SHARED_OUTPUTS}"

echo
echo "Updated ${ENV_FILE} with:"
echo "  CSD_CACHE_ROOT=\"${SHARED_CACHE}\""
echo "  CSD_OUTPUTS_ROOT=\"${SHARED_OUTPUTS}\""
echo
echo "Verify:"
echo "  source synthesis/.env"
echo "  python3 -c \"from synthesis.storage_env import ensure_shared_storage_env; ensure_shared_storage_env(); import os; print('cache', os.environ['CSD_CACHE_ROOT']); print('generated', os.environ['CSD_OUTPUT_DIR'])\""
