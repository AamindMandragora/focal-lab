#!/usr/bin/env bash
# Clone upstream baseline repositories into legacy/ (gitignored except legacy/README.md).
#
# Usage (from repo root):
#   bash environment/clone_legacy_csds.sh
#
# Environment overrides (optional):
#   LEGACY_ROOT           Destination root (default: <repo>/legacy)
#   LEGACY_SHALLOW=0      full git history (default: 1 = shallow clone)
#   LEGACY_CRANE_URL      CRANE git URL
#   LEGACY_ITERGEN_URL    IterGen git URL
#   LEGACY_CARS_URL       CARS git URL
#   LEGACY_CRANE_REF      branch/tag/commit for CRANE (passed to clone -b when shallow ok)
#   LEGACY_ITERGEN_REF    same for IterGen
#   LEGACY_CARS_REF       same for cars
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LEGACY_ROOT="${LEGACY_ROOT:-${REPO_ROOT}/legacy}"
PATCH_ROOT="${LEGACY_PATCH_ROOT:-${REPO_ROOT}/environment/legacy_patches}"
REPOS_MANIFEST="${REPO_ROOT}/environment/legacy/repos.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${LEGACY_ROOT}"

manifest_value() {
  "$PYTHON_BIN" - "$REPOS_MANIFEST" "$1" "$2" <<'PY'
import json
import sys

manifest, name, field = sys.argv[1:]
print(json.load(open(manifest))[name][field])
PY
}

CRANE_URL="${LEGACY_CRANE_URL:-$(manifest_value CRANE git_url)}"
ITERGEN_URL="${LEGACY_ITERGEN_URL:-$(manifest_value itergen git_url)}"
CARS_URL="${LEGACY_CARS_URL:-$(manifest_value cars git_url)}"
LEGACY_CRANE_REF="${LEGACY_CRANE_REF:-$(manifest_value CRANE commit)}"
LEGACY_ITERGEN_REF="${LEGACY_ITERGEN_REF:-$(manifest_value itergen commit)}"
LEGACY_CARS_REF="${LEGACY_CARS_REF:-$(manifest_value cars commit)}"
export LEGACY_CRANE_REF LEGACY_ITERGEN_REF LEGACY_CARS_REF

FETCH_ARGS=()
if [[ "${LEGACY_SHALLOW:-1}" != "0" ]]; then
  FETCH_ARGS+=(--depth 1)
fi

fetch_ref() {
  local dest="$1" ref="$2"
  if [[ ${#FETCH_ARGS[@]} -gt 0 ]]; then
    git -C "${dest}" fetch "${FETCH_ARGS[@]}" origin "${ref}"
  else
    git -C "${dest}" fetch origin "${ref}"
  fi
}

clone_repo() {
  local name="$1" url="$2" ref_var="$3"
  local dest="${LEGACY_ROOT}/${name}"
  local ref="${!ref_var:-}"

  if [[ -d "${dest}/.git" ]]; then
    echo "[legacy] existing repo: ${dest}"
    git -C "${dest}" remote set-url origin "${url}" || true
    fetch_ref "${dest}" "${ref}"
    if [[ "$(git -C "${dest}" rev-parse HEAD)" != "$(git -C "${dest}" rev-parse FETCH_HEAD)" ]]; then
      git -C "${dest}" checkout --detach --force FETCH_HEAD
      rm -rf "${dest}/.git/vas-applied-patches"
    fi
  else
    git init "${dest}"
    git -C "${dest}" remote add origin "${url}"
    fetch_ref "${dest}" "${ref}"
    git -C "${dest}" checkout --detach FETCH_HEAD
  fi

  apply_patches "${name}" "${dest}"
  local expected actual
  expected="$(git -C "${dest}" rev-parse FETCH_HEAD)"
  actual="$(git -C "${dest}" rev-parse HEAD)"
  if [[ "$actual" != "$expected" ]]; then
    echo "[legacy] ERROR: ${name} HEAD ${actual} does not match pinned ${expected}" >&2
    return 1
  fi
  echo "[legacy] verified ${name} commit ${actual}"
}

apply_patches() {
  local name="$1"
  local dest="$2"
  local pdir="${PATCH_ROOT}/${name}"
  if [[ ! -d "${pdir}" ]]; then
    return 0
  fi
  shopt -s nullglob
  local patches=( "${pdir}"/*.patch )
  shopt -u nullglob
  if [[ ${#patches[@]} -eq 0 ]]; then
    return 0
  fi
  echo "[legacy] applying ${#patches[@]} patch(es) in ${name}"
  for p in "${patches[@]}"; do
    local stamp_dir stamp
    stamp_dir="${dest}/.git/vas-applied-patches"
    stamp="${stamp_dir}/$(basename "${p}")"
    if [[ -f "$stamp" ]]; then
      echo "[legacy] already applied: $(basename "${p}")"
      continue
    fi
    if [[ -d "${dest}/.git" ]]; then
      if git -C "${dest}" apply --check "${p}"; then
        git -C "${dest}" apply "${p}"
      elif git -C "${dest}" apply --reverse --check "${p}"; then
        echo "[legacy] detected existing patch: $(basename "${p}")"
      else
        echo "[legacy] ERROR: patch does not apply cleanly: ${p}" >&2
        return 1
      fi
      mkdir -p "$stamp_dir"
      : > "$stamp"
    else
      patch -d "${dest}" -p1 --forward < "${p}" || {
        echo "[legacy] ERROR: patch failed for ${p}" >&2
        return 1
      }
    fi
  done
}

clone_repo "CRANE" "${CRANE_URL}" LEGACY_CRANE_REF
clone_repo "itergen" "${ITERGEN_URL}" LEGACY_ITERGEN_REF
clone_repo "cars" "${CARS_URL}" LEGACY_CARS_REF

echo "[legacy] done. Trees:"
ls -la "${LEGACY_ROOT}"
