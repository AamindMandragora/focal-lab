# Shared path setup for experiment scripts under experiments/scripts/.
# shellcheck disable=SC2034
if [[ -z "${METADECODE_ROOT:-}" ]]; then
  METADECODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
ROOT="$METADECODE_ROOT"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

SPLITS_DIR="${ROOT}/experiments/splits"
ENV_SPLITS_DIR="${ROOT}/environment/benchmark_splits"
WARMSTARTS_DIR="${ROOT}/experiments/warmstarts"

CONDA_ENV="${METADECODE_CONDA_ENV:-${VAS_CONDA_ENV:-/apps/conda/advayth2/envs/advayth2}}"
PY="${METADECODE_PYTHON:-${CONDA_ENV}/bin/python}"

if [[ -d "${CONDA_ENV}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
fi

ITERGEN_ROOT="${ITERGEN_ROOT:-${ROOT}/legacy/itergen}"

# IterGen-native Spider eval scripts historically used /opt/anaconda; fall back to PY.
if [[ -x /opt/anaconda/bin/python ]]; then
  ITERGEN_NATIVE_PY="${ITERGEN_NATIVE_PY:-/opt/anaconda/bin/python}"
else
  ITERGEN_NATIVE_PY="${ITERGEN_NATIVE_PY:-$PY}"
fi
