#!/bin/bash
# PURE RE-EVAL (zero Bedrock) — v4, IDENTICAL to v3 (symbol-boundary grounding) EXCEPT
# CSD_RECURRENCE_FLAT=1: the recurrence penalty is now applied FLAT (ln(0.3) once per
# distinct tried token, IterGen-faithful) instead of cumulative (ln(0.3)*count).
#
# PURPOSE: settle whether the v3 recovery NEEDS our stronger cumulative penalty, or
# whether a faithful IterGen ×0.3 (applied once) already flips the bad name within the
# cheap per-symbol retry budget.
#   - If recovery STILL happens (concert_singer -> concert, 'unit fully grounded' after a
#     penalize) -> we are genuinely IterGen-aligned on the penalty too.
#   - If it does NOT (counts climb to the budget cap with no flip) -> the recovery depends
#     on the cumulative penalty (stronger than IterGen's flat 0.3). User has pre-ruled that
#     is acceptable/fair; this run just tells us which case we're in.
#
# Compare directly against v3 (cumulative): spider1p5b_symbolboundary_v3_reeval_20260616.
# Watch: '[recurrence] penalty mode=flat(itergen)' at startup; then whether 'unit fully
# grounded' appears AFTER a penalize sequence on the concert_singer example.
#
# Eval = Qwen2.5-1.5B-Instruct via vLLM on GPU 2, 0.20 util. NOT a recorded result.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_RECURRENCE_PENALTY=0.3
export CSD_RECURRENCE_FLAT=1
export CSD_GROUNDING_LOG=1
OUT=outputs/generated/spider1p5b_flatpenalty_v4_reeval_20260616
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
STRAT=/home/aadivyar/csd-generation/grounding_fixture_strategy_20260615.dfy

CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 1 \
  --initial-strategy-file "$STRAT" \
  --output-name spider1p5b_flatpenalty_v4_reeval_20260616 \
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
echo "DONE_GROUNDING_REEVAL_V4 $(date)" | tee -a "$OUT/run.log"
