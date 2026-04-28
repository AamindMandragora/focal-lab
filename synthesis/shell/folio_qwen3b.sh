#!/bin/bash
# Make CSD for FOLIO using GPT 5.4 generation and Qwen 7B evaluation.
# Usage: bash synthesis/shell/folio_qwen3b.sh [extra generate_csd args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
csd_run_preset folio "FOLIO" "$@"
