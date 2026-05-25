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
#   LEGACY_CRANE_REF      branch/tag/commit for CRANE (passed to clone -b when shallow ok)
#   LEGACY_ITERGEN_REF    same for IterGen
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LEGACY_ROOT="${LEGACY_ROOT:-${REPO_ROOT}/legacy}"
PATCH_ROOT="${REPO_ROOT}/environment/legacy_patches"

mkdir -p "${LEGACY_ROOT}"

CRANE_URL="${LEGACY_CRANE_URL:-https://github.com/uiuc-focal-lab/CRANE.git}"
ITERGEN_URL="${LEGACY_ITERGEN_URL:-https://github.com/structuredllm/itergen.git}"
SHALLOW_ARGS=()
if [[ "${LEGACY_SHALLOW:-1}" != "0" ]]; then
  SHALLOW_ARGS+=(--depth 1)
fi

clone_repo() {
  local name="$1" url="$2" ref_var="$3"
  local dest="${LEGACY_ROOT}/${name}"
  local ref="${!ref_var:-}"

  if [[ -d "${dest}/.git" ]]; then
    echo "[legacy] existing repo: ${dest}"
    git -C "${dest}" remote set-url origin "${url}" || true
    git -C "${dest}" fetch origin --tags || git -C "${dest}" fetch origin || true
    if [[ -n "${ref}" ]]; then
      git -C "${dest}" checkout "${ref}" || {
        echo "[legacy] WARN: checkout '${ref}' failed in ${dest}; staying on current HEAD" >&2
      }
    else
      git -C "${dest}" pull --ff-only || true
    fi
  else
    if [[ -n "${ref}" ]]; then
      git clone "${SHALLOW_ARGS[@]}" --branch "${ref}" "${url}" "${dest}"
    else
      git clone "${SHALLOW_ARGS[@]}" "${url}" "${dest}"
    fi
  fi

  apply_patches "${name}" "${dest}"
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
    if [[ -d "${dest}/.git" ]]; then
      git -C "${dest}" apply "${p}" || git -C "${dest}" apply --reject --whitespace=fix "${p}" || {
        echo "[legacy] git apply failed for $(basename "${p}"); trying patch -p1" >&2
        patch -d "${dest}" -p1 --forward < "${p}" || {
          echo "[legacy] ERROR: failed to apply ${p}" >&2
          return 1
        }
      }
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

echo "[legacy] done. Trees:"
ls -la "${LEGACY_ROOT}"
