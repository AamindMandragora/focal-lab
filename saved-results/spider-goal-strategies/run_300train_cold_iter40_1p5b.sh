#!/usr/bin/env bash
# 1.5B — COLD synthesis iterating DIRECTLY on the full 300-TRAIN set (seed334), iter40.
# Rationale: the 50-set H8 winner overfit (76% on 50 -> 52.7% on 300-train). Iterating on all 300
# gives the author the full-distribution eval signal, so the format-steering lever it discovers is
# fit to 300, not 50. After this run, its best strategy is re-evaluated ONCE on the disjoint 300-TEST
# split (held-out, author never sees it) to get the true campaign number (win bar >=159/300 = 53%).
#
# IDENTICAL fair recipe to run_iter50_tok0_h8_1p5b.sh except: eval-sample-size 50 -> 300, split-name
# train on the 300x300 file, accept bars. COLD (no --initial-strategy-file), mask ON + bandit, token-0.
# Author = UIUC focal lab AWS account 887730490125 (AWS_BEARER_TOKEN_BEDROCK, us-east-1), user-approved.
#
# Accept gate: min-accuracy 0.60 (180/300). Set ABOVE the 53% win bar on purpose so the loop keeps
# iterating and builds margin for the train->test gap; if 0.60 is never reached it runs all 40 and the
# best attempt is written to failure_report.json (still usable for the held-out re-eval).
set -u
cd ~/csd-generation
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=2
SPLIT=environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider1p5b_300train_cold_iter40_${STAMP}
LOG=logs/${NAME}.log
echo "[launch] $NAME -> $LOG (author=bedrock sonnet-4.6, COLD, token-0, 300-TRAIN, bar 0.60/0.92, iter40)" | tee -a logs/300train_1p5b_driver_${STAMP}.log
python -m synthesis.run_synthesis \
  --task "$TASK" --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 40 \
  --min-accuracy 0.60 --min-syntax-rate 0.92 --no-require-delimiters \
  --eval-sample-size 300 --eval-min-examples-before-threshold-stop 300 \
  --eval-max-steps 200 --eval-step-token-budget 1 --eval-max-seconds-per-example 450 \
  --restart-after-stuck-iters 0 \
  --vllm-max-model-len 16384 \
  --vllm-gpu-memory-utilization 0.30 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --spider-split-file "$SPLIT" --spider-split-name train \
  --output-name "$NAME" \
  > "$LOG" 2>&1
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/300train_1p5b_driver_${STAMP}.log
