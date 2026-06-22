#!/bin/bash
# Pure re-eval of att16 strategy body on seed123 TRAIN split, then EVAL split.
# Purpose: check att16 (acc .510/syn .755 under old metric) against the current,
# aligned evaluator — no synthesis, just one pass each side.
# Do NOT set min-accuracy / min-syntax-rate gates (both 0.0) — we want raw numbers.
set -uo pipefail
cd /home/aadivyar/csd-generation

# Load .env credentials (AWS_BEARER_TOKEN_BEDROCK etc.)
set -a; source /home/aadivyar/csd-generation/.env; set +a

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

GSPLIT=environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
BODY=gsm1p5b_seed123_att16_body.dfy

# ── TRAIN SIDE ──────────────────────────────────────────────────────────────
echo "TRAIN_REEVAL_START $(date)"

/apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters." \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 1 \
  --initial-strategy-file "$BODY" \
  --output-name gsm1p5b_seed123_att16_aligned_train_20260612 \
  --output-dir "outputs/generated/gsm1p5b_seed123_att16_aligned_train_20260612" \
  --min-accuracy 0.0 --min-syntax-rate 0.0 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 \
  --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.40 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --refinement-beam-size 2 \
  --gsm-split-file "$GSPLIT" --gsm-split-name train

TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"

# ── EVAL (HELD-OUT) SIDE ─────────────────────────────────────────────────────
echo "HELDOUT_REEVAL_START $(date)"

/apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters." \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 1 \
  --initial-strategy-file "$BODY" \
  --output-name gsm1p5b_seed123_att16_aligned_heldout_20260612 \
  --output-dir "outputs/generated/gsm1p5b_seed123_att16_aligned_heldout_20260612" \
  --min-accuracy 0.0 --min-syntax-rate 0.0 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 \
  --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.40 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --refinement-beam-size 2 \
  --gsm-split-file "$GSPLIT" --gsm-split-name eval

HELDOUT_EXIT=$?
echo "HELDOUT_EXIT=$HELDOUT_EXIT"

echo "DONE_ATT16_ALIGNED_REEVAL"
