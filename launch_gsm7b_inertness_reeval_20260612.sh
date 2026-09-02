#!/bin/bash
# Inertness re-eval: GSM-7B controlled winner (seed123 eval, expected 69.4% acc / 93.9% syn).
# Purpose: verify the global _forbidden_allow_mask fix (blocks {, }, ** tokens) is inert
# for GSM-7B -- no accuracy drop, no unexpected syntax change.
# Strategy: gsm7b_seed123_fresh_win_body.dfy (extracted from gsm7b_seed123_fresh_20260610 run)
# Split: gsm_symbolic_crane_proportional_49x49_seed123.json, eval side, N=49
# Expected: ~69.4% acc / ~93.9% syn (under CRANE-aligned metric)
# GPU: 1 only (23 GB free; 0.45 util claims ~18.4 GB)
set -uo pipefail
cd /home/aadivyar/csd-generation

set -a; source /home/aadivyar/csd-generation/.env; set +a

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

GSPLIT=environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
BODY=/home/aadivyar/csd-generation/gsm7b_seed123_fresh_win_body.dfy
OUT=outputs/generated/gsm7b_inertness_reeval_20260612
mkdir -p "$OUT"

echo "GSM7B_INERTNESS_START $(date)"
echo "Strategy: $BODY"
echo "Split: $GSPLIT (eval side, N=49)"

/apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters." \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 1 \
  --initial-strategy-file "$BODY" \
  --output-name gsm7b_inertness_reeval_20260612 \
  --output-dir "$OUT" \
  --min-accuracy 0.0 --min-syntax-rate 0.0 \
  --eval-sample-size 49 \
  --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 \
  --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.45 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --refinement-beam-size 2 \
  --gsm-split-file "$GSPLIT" --gsm-split-name eval \
  2>&1 | tee /tmp/gsm7b_inertness_reeval_20260612.log

EXIT=$?
echo "GSM7B_INERTNESS_EXIT=$EXIT"
echo "DONE_GSM7B_INERTNESS $(date)"
