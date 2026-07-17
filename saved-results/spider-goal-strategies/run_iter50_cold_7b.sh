#!/usr/bin/env bash
# FAST-ITERATION COLD synthesis — 7B, on the 50-example proportional subset of seed334 TRAIN.
# Same fair recipe as run_cycle1_cold_7b.sh (token-0, --no-require-delimiters, bare task, COLD,
# Bedrock Sonnet-4.6 author, mask ON + bandit, --eval-max-steps 200 = IterGen budget) — the ONLY
# changes are the split (50-set), sample size (50), and the accept bar = IterGen-on-50 promotion bar.
#   - SPLIT  = spider_dev_proportional_50train_seed334.json, --spider-split-name train  (the 50)
#   - --eval-sample-size 50, threshold-stop after all 50
#   - --min-accuracy 0.76  (IterGen-on-50 = 34/50=68%; bar = beat by >=4 => 38/50=76%; see itergen-on-50-bar.md)
# COLD: NO --initial-strategy-file (warm starts permanently banned). Author = Bedrock UIUC focal lab account (AWS 887730490125)
# (AWS_BEARER_TOKEN_BEDROCK, us-east-1), user-approved spend. Framework = the 2026-06-23 fairness-fixed
# build (A1 scalar / A2 marginals / A3 pareto seed in feedback_loop.py; B task-guidance removed from prompts.py).
# Stage 2 (promote a bar-clearing candidate to full 300-train) and stage 3 (held-out 300-test) are SEPARATE,
# manual steps — this script only runs the fast 50-example iterate loop.
set -u
cd ~/csd-generation
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=1

SPLIT=environment/benchmark_splits/spider_dev_proportional_50train_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider7b_iter50_cold_${STAMP}
LOG=logs/${NAME}.log

echo "[launch] $NAME -> $LOG  (author=bedrock sonnet-4.6, COLD, token-0, 50-set train, bar 0.76)" | tee -a logs/iter50_driver_${STAMP}.log
python -m synthesis.run_synthesis \
  --task "$TASK" --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --min-accuracy 0.76 --min-syntax-rate 0.85 --no-require-delimiters \
  --eval-sample-size 50 --eval-min-examples-before-threshold-stop 50 \
  --eval-max-steps 200 --eval-step-token-budget 1 --eval-max-seconds-per-example 450 \
  --restart-after-stuck-iters 0 \
  --vllm-max-model-len 16384 \
  --vllm-gpu-memory-utilization 0.55 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --spider-split-file "$SPLIT" --spider-split-name train \
  --output-name "$NAME" \
  > "$LOG" 2>&1
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/iter50_driver_${STAMP}.log
