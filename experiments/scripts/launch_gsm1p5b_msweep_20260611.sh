#!/usr/bin/env bash
# Reproduce the 2026-06-11 GSM-1.5B max-steps pure re-evaluation sweep.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STRATEGY="$REPO_ROOT/experiments/warmstarts/gsm1p5b_seed429_warmstart_body.dfy"
GSM_SPLIT="$REPO_ROOT/experiments/splits/gsm_symbolic_crane_proportional_49x49_seed429.json"
PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${GPU:-0}
EVAL_MODEL=${EVAL_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}

run_one() {
  local max_steps=$1
  local name="gsm1p5b_seed429_ablate_tb1_ms${max_steps}_20260611"
  local output="$REPO_ROOT/outputs/generated/$name"

  echo "[max-steps-sweep] starting max_steps=$max_steps output=$output"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m synthesis.run_synthesis \
    --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
    --dataset gsm_symbolic \
    --max-iterations 1 \
    --initial-strategy-file "$STRATEGY" \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "$EVAL_MODEL" \
    --eval-backend vllm \
    --output-name "$name" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 49 \
    --eval-max-steps "$max_steps" \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 90 \
    --eval-min-examples-before-threshold-stop 49 \
    --vllm-gpu-memory-utilization 0.40 \
    --device auto \
    --output-dir "$output" \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --anthropic-thinking enabled \
    --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --vllm-tensor-parallel-size 1 \
    --gsm-split-file "$GSM_SPLIT" \
    --gsm-split-name eval
}

cd "$REPO_ROOT"
for max_steps in 64 128 192 256 384 512 700 900 1024; do
  run_one "$max_steps"
done
