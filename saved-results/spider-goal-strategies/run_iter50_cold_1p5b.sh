#!/usr/bin/env bash
# FAST-ITERATION COLD synthesis — 1.5B, on the 50-example proportional subset of seed334 TRAIN.
# Identical fair recipe to run_iter50_cold_7b.sh except eval model = Qwen2.5-1.5B-Instruct, the accept
# bar, and GPU/mem (smaller model). token-0, --no-require-delimiters, bare task, COLD, Bedrock
# Sonnet-4.6 author, mask ON + bandit, --eval-max-steps 200 (= IterGen budget). Framework = the
# 2026-06-23 fairness-fixed build.
#   - SPLIT  = spider_dev_proportional_50train_seed334.json, --spider-split-name train  (the 50)
#   - --eval-sample-size 50
#   - --min-accuracy 0.66  (IterGen-on-50 = 29/50=58%; bar = beat by >=4 => 33/50=66%; see itergen-on-50-bar.md)
# Author = Bedrock UIUC focal lab account (AWS 887730490125) (AWS_BEARER_TOKEN_BEDROCK, us-east-1), user-approved spend.
# Runs CONCURRENTLY with the 7B (this on GPU 2, 7B on GPU 1) for fast turnaround.
set -u
cd ~/csd-generation
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=2

SPLIT=environment/benchmark_splits/spider_dev_proportional_50train_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider1p5b_iter50_cold_${STAMP}
LOG=logs/${NAME}.log

echo "[launch] $NAME -> $LOG  (author=bedrock sonnet-4.6, COLD, token-0, 50-set train, bar 0.66)" | tee -a logs/iter50_driver_${STAMP}.log
python -m synthesis.run_synthesis \
  --task "$TASK" --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --min-accuracy 0.66 --min-syntax-rate 0.85 --no-require-delimiters \
  --eval-sample-size 50 --eval-min-examples-before-threshold-stop 50 \
  --eval-max-steps 200 --eval-step-token-budget 1 --eval-max-seconds-per-example 450 \
  --restart-after-stuck-iters 0 \
  --vllm-max-model-len 16384 \
  --vllm-gpu-memory-utilization 0.30 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --spider-split-file "$SPLIT" --spider-split-name train \
  --output-name "$NAME" \
  > "$LOG" 2>&1
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/iter50_driver_${STAMP}.log
