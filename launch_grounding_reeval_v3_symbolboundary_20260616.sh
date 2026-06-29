#!/bin/bash
# PURE RE-EVAL (NOT synthesis, ZERO Bedrock author cost) — v3, FIRST run with the
# SYMBOL-BOUNDARY grounding mechanism active (IterGen SymbolPosMap port).
#
# What changed since v2 (deep-penalty-only, which fired but could not RECOVER):
#   - Vendored IncrementalParser now records table_ref/column_ref completions in a
#     SymbolPosMap side-record (T0-proven byte-identical accept-sets).
#   - New Parser extern CompletedSchemaSymbolCount(prefix) reads that count.
#   - RegenerateUnitOnGroundingFailure now grounds at the SCHEMA-SYMBOL boundary
#     (count rises) instead of whole-query IsCompletePrefix, with per-symbol
#     checkpoints -> rollbacks are CHEAP (replay ~one symbol, not the 38-token
#     query), so the x0.3 recurrence penalty can ACCUMULATE and flip a bad name.
#
# --initial-strategy-file supplies the hand-written grounding fixture (verifies:
# dafny 171/0 library + fixture). --max-iterations 1 with bars at 0 => the author
# is NEVER called (no refinement) => zero Bedrock. This is the documented pure
# re-eval path (warm-start ban allows --initial-strategy-file only for max-iter-1).
#
# SUCCESS SIGNAL (vs v2): we now expect, on at least one example,
#   [grounding] first-ungrounded token_idx=K of N      (a bad ident located)
#   [recurrence] penalize ... at prefix_len=P  (P>1)    (penalized at its deep pos)
#   ...repeated CHEAP retries on the SAME symbol (penalty accumulates)...
#   [grounding] unit fully grounded (n=... tokens)      (the name FLIPPED -> recovery)
# i.e. grounded_True > 0, which v2 never reached. NOT a recorded result.
#
# Eval = Qwen2.5-1.5B-Instruct via vLLM on GPU 1, 0.20 util. CSD_RECURRENCE_PENALTY=0.3
# (matches IterGen). 60-example sample.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_RECURRENCE_PENALTY=0.3
export CSD_GROUNDING_LOG=1
OUT=outputs/generated/spider1p5b_symbolboundary_v3_reeval_20260616
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
  --output-name spider1p5b_symbolboundary_v3_reeval_20260616 \
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
echo "DONE_GROUNDING_REEVAL_V3 $(date)" | tee -a "$OUT/run.log"
