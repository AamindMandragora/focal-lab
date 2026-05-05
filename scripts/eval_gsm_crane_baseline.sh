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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
DAFNY_PATH_ARG=("--dafny-path" "${DAFNY_PATH:-/home/aadivyar/.dotnet/tools/dafny}")

cp dafny/CraneCSD.dfy dafny/GeneratedCSD.dfy

python run_synthesis.py \
  --task "Solve math word problems step by step, writing each arithmetic computation inside << >> delimiters." \
  --dataset gsm_symbolic \
  --compile-only \
  --output-name generated_csd \
  "${DAFNY_PATH_ARG[@]}" \
  --device cuda \
  --eval-backend vllm \
  --generation-backend vllm \
  --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --generation-model "Qwen/Qwen2.5-Coder-7B-Instruct"

RUN_DIR="$(tr -d '\r' < outputs/generated-csd/latest_run.txt)"
python -m evaluations.gsm_symbolic.cli \
  --run-dir "$RUN_DIR" \
  --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --device cuda \
  --limit 50 \
  --max-steps 1024
