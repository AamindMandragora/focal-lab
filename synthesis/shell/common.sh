#!/bin/bash
# Shared shell helpers for CSD synthesis wrappers.

set -euo pipefail

csd_shell_setup() {
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  cd "$REPO_ROOT"

  if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
  fi

  DEVICE="${DEVICE:-auto}"
  MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
  MAX_TOKENS="${MAX_TOKENS:-1200}"
  GENERATION_TIMEOUT="${GENERATION_TIMEOUT:-0}"
  TEMPERATURE="${TEMPERATURE:-0.7}"
  MODEL_PRESET="${MODEL_PRESET:-gpt54}"
  EVAL_MODEL_PRESET="${EVAL_MODEL_PRESET:-qwen7b}"

  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf-datasets-local}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
  export CSD_HELPER_REFERENCE_MODE="${CSD_HELPER_REFERENCE_MODE:-curated}"
  export CSD_EVAL_CPU_FALLBACK="${CSD_EVAL_CPU_FALLBACK:-0}"
  mkdir -p "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"
}

csd_require_generation_credentials() {
  for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
      return 0
    fi
  done
  if [[ "$MODEL_PRESET" == gpt* && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set. Export it in your shell before running GPT-backed generation." >&2
    exit 2
  fi
}

csd_run_preset() {
  local dataset="$1"
  local label="$2"
  shift 2

  csd_shell_setup
  csd_require_generation_credentials "$@"

  echo "Making $label CSD ($MODEL_PRESET generation, $EVAL_MODEL_PRESET evaluation) from the preset wrapper..."
  echo "Offline defaults: HF_HUB_OFFLINE=$HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE, HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE."
  echo "Helper reference mode: $CSD_HELPER_REFERENCE_MODE."
  echo "CPU fallback for evaluation: $CSD_EVAL_CPU_FALLBACK."
  echo "Tip: set DEVICE=cuda:3 or CUDA_VISIBLE_DEVICES=3 to pin Qwen evaluation."
  echo ""

  python -m synthesis.cli.generate_csd "$dataset" \
    --model-preset "$MODEL_PRESET" \
    --eval-model-preset "$EVAL_MODEL_PRESET" \
    --max-iterations "$MAX_ITERATIONS" \
    --temperature "$TEMPERATURE" \
    --device "$DEVICE" \
    --max-tokens "$MAX_TOKENS" \
    --generation-timeout "$GENERATION_TIMEOUT" \
    "$@"

  echo "$label CSD done. Run dir: outputs/ (see latest_run.txt)"
}
