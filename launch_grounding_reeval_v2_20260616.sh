#!/bin/bash
# PURE RE-EVAL (NOT synthesis, ZERO Bedrock author cost): confirm the SYMBOL-BOUNDARY
# grounding rollback now penalizes the OUT-OF-SCHEMA identifier's OWN token (not the
# query's first token), and that the rollback RECOVERS (a later unit grounds).
#
# --initial-strategy-file supplies the hand-written grounding fixture that VERIFIES
# (dafny: 2 verified, 0 errors against the updated library), and --max-iterations 1
# with bars at 0 means the author is NEVER called (no refinement) -> zero Bedrock.
# This is the documented pure-re-eval path (warm-start ban allows --initial-strategy-file
# only for max-iterations-1 re-eval).
#
# What we watch for in the log (CSD_GROUNDING_LOG=1 attaches an INFO stderr handler):
#   [grounding] first-ungrounded token_idx=K of N   -> NEW extern located a bad ident
#                                                       at token K; K>0 proves it is NOT
#                                                       penalizing the query's 1st token
#   [recurrence] penalize subset_idx=... at prefix_len=P  -> penalty recorded at the bad
#                                                       token's DEEP position (P>1)
#   [grounding] unit fully grounded (n=... tokens)   -> a unit grounded (recovery, or a
#                                                       clean unit) -> grounded_True>0
# The success signal vs the OLD broken behavior: penalize lines at prefix_len>1 (was
# always prefix_len=1) AND at least one "unit fully grounded" after a penalize on the
# same example (the deep name was corrected). NOT a recorded result.
#
# Eval = Qwen2.5-1.5B-Instruct via vLLM on GPU 1, 0.20 util. CSD_RECURRENCE_PENALTY=0.3
# (matches IterGen). 60-example sample.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_RECURRENCE_PENALTY=0.3
export CSD_GROUNDING_LOG=1
OUT=outputs/generated/spider1p5b_groundfix_v2_reeval_20260616
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
  --output-name spider1p5b_groundfix_v2_reeval_20260616 \
  --min-accuracy 0.0 --min-syntax-rate 0.0 \
  --eval-sample-size 60 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 300 --eval-min-examples-before-threshold-stop 60 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  > "$OUT/run.log" 2>&1
echo "REEVAL_EXIT=$?" | tee -a "$OUT/run.log"
echo "DONE_GROUNDING_REEVAL_V2 $(date)" | tee -a "$OUT/run.log"
