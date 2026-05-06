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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
DAFNY_PATH_ARG=("--dafny-path" "${DAFNY_PATH:-/home/aadivyar/.dotnet/tools/dafny}")

python run_synthesis.py \
  --task "Solve GSM-Symbolic math word problems. Natural-language reasoning may stay outside constrained spans; for scoring, the required constrained region is the final answer only. The last <<...>> span must contain one valid symbolic arithmetic expression that answers the question. Do not require every intermediate calculation to be wrapped." \
  --dataset gsm_symbolic \
  --max-iterations 15 \
  --generation-model "gpt-5.4" \
  --generation-backend openai \
  --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --eval-backend vllm \
  --output-name "gsm_new_tools_csd" \
  --temperature 0.7 \
  --synthesis-max-tokens 1536 \
  "${DAFNY_PATH_ARG[@]}" \
  --device cuda \
  --min-accuracy 0.3 \
  --min-syntax-rate 1.0 \
  --eval-sample-size 10 \
  --eval-max-steps 900 \
  --eval-seed 123
