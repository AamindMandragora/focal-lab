#!/bin/bash
# Make CSD for GSM-Symbolic (math reasoning) using GPT 5.4 generation and Qwen 7B evaluation.
# Usage: bash synthesis/shell/gsm_symbolic_qwen7b.sh [extra generate_csd args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
export CSD_REQUIRE_NATURAL_DELIMITERS="${CSD_REQUIRE_NATURAL_DELIMITERS:-1}"
source "$SCRIPT_DIR/common.sh"
csd_run_preset gsm_symbolic "GSM-Symbolic" "$@"
