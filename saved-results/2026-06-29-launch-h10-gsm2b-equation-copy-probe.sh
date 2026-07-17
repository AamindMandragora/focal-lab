#!/usr/bin/env bash
set -uo pipefail

# H10 GSM-2B equation-copy mechanism probe.
# This is a pure local re-eval: --max-iterations 1 + --initial-strategy-file.
# It must not call a paid author model. Keep cloud/billing env vars unset.

REPO=/home/aadivyar/csd-generation
PY=/apps/conda/aadivyar/envs/csd/bin/python
BODY="$REPO/saved-results/2026-06-29-h10-gsm2b-equation-copy-probe-body.dfy"
OUT_NAME=h10_gsm2b_equation_copy_probe_20260629
OUT="$REPO/outputs/generated/$OUT_NAME"
LOG="$OUT/run.log"
SPLIT="$REPO/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export LD_LIBRARY_PATH=/apps/conda/aadivyar/envs/csd/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/home/aadivyar/.cache/huggingface
export TRANSFORMERS_CACHE=/home/aadivyar/.cache/huggingface

unset AWS_BEARER_TOKEN_BEDROCK
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
unset AWS_PROFILE
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY

cd "$REPO" || exit 2
mkdir -p "$OUT"

echo "H10_START out=$OUT body=$BODY gpu=$CUDA_VISIBLE_DEVICES $(date -Is)" | tee "$OUT/launch.log"

"$PY" -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --max-iterations 1 \
  --initial-strategy-file "$BODY" \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --generation-backend bedrock \
  --eval-model Qwen/Qwen3.5-2B \
  --eval-backend vllm \
  --output-name "$OUT_NAME" \
  --output-dir "$OUT" \
  --min-accuracy 0.0 \
  --min-syntax-rate 0.0 \
  --eval-sample-size 49 \
  --eval-max-steps 900 \
  --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 600 \
  --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.25 \
  --device auto \
  --adaptive-helper-mask \
  --helper-selection-policy bandit \
  --refinement-beam-size 2 \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file "$SPLIT" \
  --gsm-split-name train \
  > "$LOG" 2>&1

ec=$?
echo "H10_DONE exit=$ec out=$OUT $(date -Is)" | tee -a "$OUT/launch.log"
exit "$ec"
