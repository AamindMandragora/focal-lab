#!/bin/bash
# Make CSD for Chem-CoT-Bench using GPT 5.4 generation and Qwen 7B evaluation.
# Usage: bash synthesis/shell/chem_cot_bench_gpt54_qwen7b.sh [extra generate_csd args]

set -euo pipefail

source "$(dirname "$0")/common.sh"

csd_run_preset chem_cot_bench "Chem-CoT-Bench" "$@"
