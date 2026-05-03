#!/bin/bash
# Make CSD for GSM-Symbolic (math reasoning) using GPT 5.4 generation and Qwen 7B evaluation.
# Usage: bash synthesis/shell/gsm_symbolic_qwen7b.sh [extra generate_csd args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAFNY_MODE=0
prev_arg=""
for arg in "$@"; do
  if [[ "$arg" == "--strategy-language=dafny" ]]; then
    DAFNY_MODE=1
    break
  fi
  if [[ "$prev_arg" == "--strategy-language" && "$arg" == "dafny" ]]; then
    DAFNY_MODE=1
    break
  fi
  prev_arg="$arg"
done

if [[ "$DAFNY_MODE" -eq 1 ]]; then
  MAX_ITERATIONS="${MAX_ITERATIONS:-30}"
  export CSD_GENERATION_SEARCH_ATTEMPTS="${CSD_GENERATION_SEARCH_ATTEMPTS:-10}"
else
  MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
fi

export CSD_REQUIRE_NATURAL_DELIMITERS="${CSD_REQUIRE_NATURAL_DELIMITERS:-1}"
export CSD_GSM_PREFER_SCRATCH_SPANS="${CSD_GSM_PREFER_SCRATCH_SPANS:-0}"
export CSD_STRICT_COMPLETE_ORDER="${CSD_STRICT_COMPLETE_ORDER:-1}"
source "$SCRIPT_DIR/common.sh"
csd_run_preset gsm_symbolic "GSM-Symbolic" "$@"
