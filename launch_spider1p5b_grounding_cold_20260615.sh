#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON, with the NEW grounding helper (2026-06-15).
#
# What changed vs launch_spider1p5b_cleanwin_cold3_20260612.sh:
#   * NEW LEVER: library helper RegenerateUnitOnGroundingFailure + lm.SpanGrounded extern.
#     The author can now fill a constrained span unit-by-unit, rewinding/resampling any
#     completed unit whose identifier-like tokens are not grounded in the prompt context.
#     It is in NON_PRUNABLE_HELPERS (always visible to the author) and carries a length
#     ensures so a strategy using it can satisfy the template's |generated|<=|prefix|+maxSteps
#     bound. A verified few-shot example ("// Grounded-unit constrained CSD.") demonstrates it.
#     Targets the 64/147 out-of-schema errors measured on the prior held-out run.
#   * MASK ON (2026-06-14 ruling): --no-adaptive-helper-mask  ->  --adaptive-helper-mask
#     --helper-selection-policy bandit. The mask is a required shared condition for every cell.
#   * Accept bars: --min-accuracy 0.55 (clean margin over IterGen 0.52) --min-syntax-rate 0.85
#     (within the 10-15pp tolerance of IterGen 0.947; --eval-min-examples-before-threshold-stop
#     300 means no early small-N cut).
# COLD start (no --initial-strategy-file). Author = Bedrock claude-sonnet-4-6, thinking high
# (work Bedrock cred AWS_BEARER_TOKEN_BEDROCK from the project .env, us-east-1).
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
OUT=outputs/generated/spider1p5b_grounding_cold_20260615
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_grounding_cold_20260615 \
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
echo "DONE_SPIDER1P5B_GROUNDING_COLD $(date)"
