#!/bin/bash
# Make CSD for Spider text-to-SQL using GPT 5.4 generation and Qwen 7B evaluation.
# Usage: bash synthesis/shell/spider_gpt54_qwen7b.sh [extra generate_csd args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_TOKENS="${MAX_TOKENS:-1000}"
source "$SCRIPT_DIR/common.sh"
csd_run_preset spider "Spider SQL" "$@"
