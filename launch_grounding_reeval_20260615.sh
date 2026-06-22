#!/bin/bash
# PURE RE-EVAL (NOT synthesis, ZERO Bedrock author cost): confirm Change 3 fires
# end-to-end. --initial-strategy-file supplies a hand-written grounding fixture that
# VERIFIES (dafny: 2 verified, 0 errors), and --max-iterations 1 with bars at 0 means
# the author is never called and no refinement is triggered (documented pure-re-eval
# path; user gate: "targeted re-eval (cheap)").
#
# What we watch for in the log (CSD_GROUNDING_LOG=1 attaches an INFO stderr handler):
#   [grounding] span=... grounded=False     -> a real grounding failure on the 1.5B
#   [recurrence] penalize subset_idx=... at  -> the rollback CALLED PenalizeTriedTokenAt
#   counts now={...}                          -> the persistent penalty bumped (Change 3)
# Seeing a [recurrence] penalize line after a grounded=False is the end-to-end proof
# that the rollback diverges instead of being the old no-op.
#
# Eval = Qwen2.5-1.5B-Instruct via vLLM on GPU 1 (~2GB used of 41GB free), 0.20 util.
# CSD_RECURRENCE_PENALTY=0.3 (matches IterGen). 60-example sample to make at least one
# grounding failure + rollback likely. NOT a recorded result.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_RECURRENCE_PENALTY=0.3
export CSD_GROUNDING_LOG=1
OUT=outputs/generated/spider1p5b_groundfix_reeval_20260615
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
STRAT=/home/aadivyar/csd-generation/grounding_fixture_strategy_20260615.dfy

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 1 \
  --initial-strategy-file "$STRAT" \
  --output-name spider1p5b_groundfix_reeval_20260615 \
  --min-accuracy 0.0 --min-syntax-rate 0.0 \
  --eval-sample-size 60 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 300 --eval-min-examples-before-threshold-stop 60 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train
echo "REEVAL_EXIT=$?"
echo "DONE_GROUNDING_REEVAL $(date)"
