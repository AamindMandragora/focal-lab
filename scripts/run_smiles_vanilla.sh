#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

LATEST_RUN_FILE="outputs/generated-csd/latest_run.txt"
if [[ ! -f "$LATEST_RUN_FILE" ]]; then
  echo "Missing $LATEST_RUN_FILE. Run scripts/run_smiles_synthesis.sh first." >&2
  exit 1
fi

RUN_DIR="$(tr -d '\r' < "$LATEST_RUN_FILE")"
if [[ -z "$RUN_DIR" ]]; then
  echo "latest_run.txt is empty." >&2
  exit 1
fi

python -m evaluations.smiles.cli \
  --run-dir "$RUN_DIR" \
  --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --device cuda \
  --limit 100 \
  --max-steps 256 \
  --seed 123 \
  --unconstrained
