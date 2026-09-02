#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON, span-close fix (2026-06-15).
#
# What changed vs launch_spider1p5b_grounding_cold_20260615.sh (the killed run):
#   * ROOT CAUSE FIXED: the prior run regressed because the spans never closed
#     (101/141 outputs opened "<<" but never reached ">>"). The grounding few-shot
#     example was buggy — it filled the span but NEVER called CloseConstrainedSpan,
#     so the author learned to leave spans open. The grounding helper was never even
#     invoked (the binding constraint was span-closure, not out-of-schema errors).
#   * NEW LIBRARY HELPER: CloseSpanWithinBudget(lm, parser, prompt, generated,
#     currentConstrained, eosToken, budget) — advances an open span to a completable
#     state (dead-end-aware), tracks the longest complete point, and emits ">>"
#     within a step budget; verified (length/cost bounds), in NON_PRUNABLE_HELPERS
#     (always visible to the author). Composes only already-verified primitives.
#   * FIXED few-shot example "// Grounded-and-closed constrained CSD." now fills with
#     RegenerateUnitOnGroundingFailure on HALF the budget, then closes with
#     CloseSpanWithinBudget on the rest. NEW example "// Open-then-reliably-close CSD."
#     demonstrates the minimal open->close pattern. Both verify under the template.
#   * Grounding helper RETAINED (the lever for out-of-schema errors, which become the
#     dominant error mode only once spans reliably close).
#   * MASK ON (2026-06-14 ruling): --adaptive-helper-mask --helper-selection-policy bandit.
#   * Accept bars: --min-accuracy 0.55 (clean margin over IterGen 0.52)
#     --min-syntax-rate 0.85 (within ~10-15pp of IterGen 0.947).
#   * GPU 1 (GPU 2 has an orphaned vLLM worker holding ~8.7GB that the safety
#     classifier blocked killing; GPU 1 is clean with ~38GB free).
# COLD start (no --initial-strategy-file). Author = Bedrock claude-sonnet-4-6, thinking
# high (WORK Bedrock cred AWS_BEARER_TOKEN_BEDROCK from the project .env, us-east-1 —
# same work cred as every prior synthesis run; NOT a personal account).
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
OUT=outputs/generated/spider1p5b_spanclose_cold_20260615
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_spanclose_cold_20260615 \
  --min-accuracy 0.55 --min-syntax-rate 0.85 \
  --eval-sample-size 300 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 300 --eval-min-examples-before-threshold-stop 300 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train
echo "SYNTH_EXIT=$?"
echo "DONE_SPIDER1P5B_SPANCLOSE_COLD $(date)"
