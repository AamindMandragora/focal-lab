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
DAFNY_PATH_ARG=("--dafny-path" "${DAFNY_PATH:-/home/aadivyar/.dotnet/tools/dafny}")

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print('Python executable:', sys.executable)"

"$PYTHON_BIN" run_synthesis.py \
  --task "Answer constrained molecular generation problems by producing the requested chemistry answer string, typically a SMILES string, inside << >> delimiters." \
  --dataset smiles \
  --max-iterations 10 \
  --generation-model "gpt-5.4" \
  --generation-backend openai \
  --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --eval-backend vllm \
  --output-name "smiles_csd" \
  --temperature 0.7 \
  --synthesis-max-tokens 1536 \
  "${DAFNY_PATH_ARG[@]}" \
  --device cuda \
  --min-accuracy 0.2 \
  --min-syntax-rate 1.0 \
  --eval-sample-size 10 \
  --eval-max-steps 256 \
  --eval-seed 123 \
  --vllm-tensor-parallel-size 1 \
  --vllm-gpu-memory-utilization 0.40
