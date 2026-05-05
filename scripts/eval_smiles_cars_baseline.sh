#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -f outputs/generated-csd/latest_run.txt ]]; then
  echo "Missing outputs/generated-csd/latest_run.txt. Run synthesis first." >&2
  exit 1
fi

RUN_DIR="$(tr -d '\r' < outputs/generated-csd/latest_run.txt)"
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "Could not resolve run dir from latest_run.txt: $RUN_DIR" >&2
  exit 1
fi

COMPILED_MODULE="$(find "$RUN_DIR" -maxdepth 3 -name GeneratedCSD.py | head -n 1)"
if [[ -z "$COMPILED_MODULE" || ! -f "$COMPILED_MODULE" ]]; then
  echo "Could not find GeneratedCSD.py under $RUN_DIR" >&2
  exit 1
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print('Python executable:', sys.executable)"
echo "Run dir: $RUN_DIR"
echo "Compiled module: $COMPILED_MODULE"

SEED="${SEED:-$(date +%s)}"
echo "Sampling seed: $SEED"

"$PYTHON_BIN" scripts/run_smiles_cars_baseline.py \
  --compiled-module "$COMPILED_MODULE" \
  --model-name "${MODEL_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}" \
  --backend "${BACKEND:-vllm}" \
  --device "${DEVICE:-cuda}" \
  --classes "${CLASSES:-acrylates,chain_extenders,isocyanates}" \
  --target-samples "${TARGET_SAMPLES:-100}" \
  --max-attempts "${MAX_ATTEMPTS:-1000}" \
  --max-steps "${MAX_STEPS:-512}" \
  --seed "$SEED"
