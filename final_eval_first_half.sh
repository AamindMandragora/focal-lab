#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CRANE_REPO="${CRANE_REPO:-$HOME/CRANE}"
export ITERGEN_REPO="${ITERGEN_REPO:-$HOME/itergen}"
export CARS_REPO="${CARS_REPO:-$HOME/cars}"
export CRANE_GSM_SYMBOLIC_DIR="${CRANE_GSM_SYMBOLIC_DIR:-$CRANE_REPO/src/gsm_symbolic}"

PYTHON_BIN="/home/advayth2/envs/vas-rdkit/bin/python"
OUTPUT_DIR="./outputs/generated-csd/"
PYTHON_ENV_DIR="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
export LD_LIBRARY_PATH="$PYTHON_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
COLLECTED_DIR="$OUTPUT_DIR/collected_benchmarks/handoff_${RUN_STAMP}"

collect_reports() {
    "$PYTHON_BIN" scripts/collect_benchmark_reports.py \
        --output-dir "$OUTPUT_DIR" \
        --dest "$COLLECTED_DIR" \
        --run-prefix "handoff_" \
        --run-stamp "$RUN_STAMP"
}

trap 'collect_reports || true; echo "[benchmarks] consolidated reports: $COLLECTED_DIR"' EXIT

run_matrix() {
    local phase="$1"
    local part="$2"
    local datasets="$3"
    local methods="$4"
    local models="$5"

    "$PYTHON_BIN" scripts/master_experiment_matrix.py \
        --run-name "handoff_${phase}_${part}_${RUN_STAMP}" \
        --output-dir "$OUTPUT_DIR" \
        --datasets "$datasets" \
        --methods "$methods" \
        --models "$models" \
        --generation-models gpt54 \
        --no-kill-vllm-before-cells \
        --no-include-ablations

    collect_reports
}

run_for_all_models() {
    local phase="$1"
    local part_prefix="$2"
    local datasets="$3"
    local methods="$4"

    run_matrix "$phase" "${part_prefix}_qwen15" "$datasets" "$methods" qwen25_coder_1p5b_instruct
    run_matrix "$phase" "${part_prefix}_qwen7" "$datasets" "$methods" qwen25_coder_7b_instruct
    run_matrix "$phase" "${part_prefix}_qwen14" "$datasets" "$methods" qwen25_coder_14b_instruct
}

# Remaining GSM/Spider cells as of handoff_20260506_235826:
# - CRANE/IterGen are complete for 1.5B, 7B, and 14B.
# - Unconstrained is complete except GSM 14B, which previously OOMed.
# - MetaDecode has not produced final GSM/Spider benchmark reports yet.
run_matrix gsm_spider remaining_unconstrained_qwen14_gsm gsm unconstrained qwen25_coder_14b_instruct
run_for_all_models gsm_spider remaining_metadecode_gpt54 gsm,spider metadecode
