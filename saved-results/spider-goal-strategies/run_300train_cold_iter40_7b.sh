#!/usr/bin/env bash
# 7B — COLD synthesis iterating DIRECTLY on the full 300-TRAIN set (seed334), iter40.
# Mirror of run_300train_cold_iter40_1p5b.sh: SAME fair recipe (COLD, mask ON + bandit, token-0,
# no grammar/grader/split edits). The ONLY differences are: eval-model 1.5B->7B, vllm util 0.30->0.45
# (7B weights ~15GB), accept bars, output name, and GPU taken from the chain via $GPU.
#
# Rationale (user ruling 2026-06-24): drop the 50-example subset entirely. Since scaling synthesis
# iterations (iter40) is the proven lever that reaches a win, iterate the author DIRECTLY on the full
# 300-train distribution (no 50->300 overfit gap), then re-evaluate the best strategy ONCE on the
# disjoint 300-TEST split (held-out, author never sees it) for the true number.
#
# Accept gate: min-accuracy 0.72 (above the believed ~66.7% = 200/300 win bar, to build train->test
# margin). If 0.72 is never reached it runs all 40 iters and the best attempt goes to failure_report.json.
# GPU: set by the chain wrapper (export GPU=<n>); defaults to 1.
# Author = UIUC focal lab AWS account 887730490125 (AWS_BEARER_TOKEN_BEDROCK, us-east-1), user-approved.
set -u
cd ~/csd-generation
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=${GPU:-1}
SPLIT=environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider7b_300train_cold_iter40_${STAMP}
LOG=logs/${NAME}.log
echo "[launch] $NAME -> $LOG (author=bedrock sonnet-4.6, COLD, token-0, 300-TRAIN, bar 0.72/0.92, iter40, GPU=$CUDA_VISIBLE_DEVICES)" | tee -a logs/300train_7b_driver_${STAMP}.log
python -m synthesis.run_synthesis \
  --task "$TASK" --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 40 \
  --min-accuracy 0.72 --min-syntax-rate 0.92 --no-require-delimiters \
  --eval-sample-size 300 --eval-min-examples-before-threshold-stop 300 \
  --eval-max-steps 200 --eval-step-token-budget 1 --eval-max-seconds-per-example 450 \
  --restart-after-stuck-iters 0 \
  --vllm-max-model-len 16384 \
  --vllm-gpu-memory-utilization 0.45 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --spider-split-file "$SPLIT" --spider-split-name train \
  --output-name "$NAME" \
  > "$LOG" 2>&1
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/300train_7b_driver_${STAMP}.log
