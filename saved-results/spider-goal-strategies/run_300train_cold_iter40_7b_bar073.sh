#!/usr/bin/env bash
# 7B — COLD synthesis iterating DIRECTLY on the full 300-TRAIN set (seed334), iter40.
# RELAUNCH 2026-06-25: copy of run_300train_cold_iter40_7b.sh with the accept bar raised 0.72 -> 0.73.
# Prior bar 0.72 was never reached (peak 69.3% train, then 9 flat iters). New bar 0.73 = IterGen TEST
# 65.67% (197/300) + the observed ~7.3pp train->test drop, so an accepted strategy is EXPECTED to clear
# IterGen on the held-out TEST split. Syntax bar 0.92 kept. Also benefits from the 2026-06-25
# delimiter-feedback leak fix (no "Contains << >>" line under --no-require-delimiters on refinement).
# GPU hardcoded to 1 (free at launch; GPU2/3 busy). COLD (no --initial-strategy-file), mask ON + bandit, token-0.
# Author = UIUC focal lab AWS account 887730490125 (AWS_BEARER_TOKEN_BEDROCK, us-east-1), user-approved.
#
# If 0.73 is never reached it runs all 40 iters and the best attempt is written to failure_report.json
# (still usable for the held-out 300-TEST re-eval).
set -u
cd ~/csd-generation
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=1
SPLIT=environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider7b_300train_cold_iter40_bar073_${STAMP}
LOG=logs/${NAME}.log
echo "[launch] $NAME -> $LOG (author=bedrock sonnet-4.6, COLD, token-0, 300-TRAIN, bar 0.73/0.92, iter40, GPU=1)" | tee -a logs/300train_7b_bar073_driver_${STAMP}.log
python -m synthesis.run_synthesis \
  --task "$TASK" --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 40 \
  --min-accuracy 0.73 --min-syntax-rate 0.92 --no-require-delimiters \
  --eval-sample-size 300 --eval-min-examples-before-threshold-stop 300 \
  --eval-max-steps 200 --eval-step-token-budget 1 --eval-max-seconds-per-example 450 \
  --restart-after-stuck-iters 0 \
  --vllm-max-model-len 16384 \
  --vllm-gpu-memory-utilization 0.45 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --spider-split-file "$SPLIT" --spider-split-name train \
  --output-name "$NAME" \
  > "$LOG" 2>&1
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/300train_7b_bar073_driver_${STAMP}.log
