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

"$PYTHON_BIN" scripts/run_experiment_part.py --output-dir "$OUTPUT_DIR" run baselines_qwen15
"$PYTHON_BIN" scripts/run_experiment_part.py --output-dir "$OUTPUT_DIR" run metadecode_gpt54_qwen15
"$PYTHON_BIN" scripts/run_experiment_part.py --output-dir "$OUTPUT_DIR" run baselines_qwen7
"$PYTHON_BIN" scripts/run_experiment_part.py --output-dir "$OUTPUT_DIR" run metadecode_gpt54_qwen7
"$PYTHON_BIN" scripts/run_experiment_part.py --output-dir "$OUTPUT_DIR" run baselines_qwen14
"$PYTHON_BIN" scripts/run_experiment_part.py --output-dir "$OUTPUT_DIR" run metadecode_gpt54_qwen14
