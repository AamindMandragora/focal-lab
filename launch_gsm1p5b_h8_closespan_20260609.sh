#!/bin/bash
# H8: reduce the proof burden. The Dafny library now has a pre-verified helper
#   helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrained) -> (generatedOut,
#   insideOut, currentOut, closed): it closes the span (delegating to the proven
#   CloseConstrainedSpan) iff parser.IsCompletePrefix holds, else is a zero-cost no-op.
#   It is registered in prompts.py (helper fence + Tool API Reference w/ usage idiom) and
#   woven into the verified "Simple delimiter-triggered CSD" example (Option A + C). It is
#   in NON_PRUNABLE_HELPERS. The author is NOT told WHEN to call it (discovery preserved).
# EXACTLY ONE conceptual variable vs the H6 baseline: the close-if-complete primitive is
#   now available + shown. The H7 feedback-hint enrichment was REVERTED to the H6 text
#   (so this is a clean single-variable test of the helper, not helper+hint). All prior
#   confirmed corrections (H4 de-bias, H5 postcondition, H6 threshold 0.1) are KEPT.
# Everything else identical to H6/H7: warmstart baseline initial strategy, --min-syntax-rate
#   0.92 (forces iteration), --min-accuracy 0.31, --max-iterations 15, Sonnet-4-6 author
#   thinking-high, seed429 TRAIN split.
set -uo pipefail
cd /home/aadivyar/csd-generation

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

OUT=outputs/generated/ralph_1p5B_gsm_h8_closespan_20260609
GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed429.json
LOG=/tmp/ralph_1p5B_gsm_h8_closespan_20260609.log

mkdir -p "$OUT"

nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_1p5B_gsm_h8_closespan_20260609 \
  --output-dir "$OUT" \
  --min-accuracy 0.31 --min-syntax-rate 0.92 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 120 --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.18 --device auto \
  --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --gsm-split-file "$GSM_SPLIT" --gsm-split-name train \
  --initial-strategy-file /home/aadivyar/csd-generation/gsm1p5b_seed429_warmstart_body.dfy \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
echo "OUT=$OUT"
