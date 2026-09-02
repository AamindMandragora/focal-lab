#!/usr/bin/env bash
# H8 (7B) — 7B, 50-example seed334-train subset. IDENTICAL fair recipe to run_iter50_tok0_7b.sh
# (same prompts.py, COLD, mask ON + bandit, token-0, bars 0.76/0.85). The ONLY scientific change vs
# run_iter50_tok0_7b.sh is --max-iterations 20 -> 40.
#
# Motivation: the 1.5B H8 (iter40) run broke a 52% plateau and reached 38/50=76% at attempt 28 by
# the author DISCOVERING a fair AppendTaskGuidance format-steering string (model-derived from gold,
# att10 CLEAN). The 7B's prior cold run peaked at 64% (32/50) within 20 iterations and was called a
# "model wall" — but that may just be too few iterations to find the same lever. H8 gives the 7B the
# same extended budget. 7B failures are the same Bucket-B wrong-structure type the lever addresses.
#
# GPU 2, util 0.45 (7B weights ~15GB; KV tiny at 485-prompt/200-gen, output-neutral). Launched by the
# chain wrapper only after the 1.5B 300-train re-eval frees GPU 2.
# Author = UIUC focal lab AWS account 887730490125 (AWS_BEARER_TOKEN_BEDROCK, us-east-1), user-approved.
set -u
cd ~/csd-generation
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=2
SPLIT=environment/benchmark_splits/spider_dev_proportional_50train_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider7b_iter50_tok0_h8_iter40_cold_${STAMP}
LOG=logs/${NAME}.log
echo "[launch] $NAME -> $LOG (author=bedrock sonnet-4.6, COLD, token-0, 50-set train, bar 0.76, H8 max-iter 40)" | tee -a logs/iter50_tok0_h8_7b_driver_${STAMP}.log
python -m synthesis.run_synthesis \
  --task "$TASK" --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 40 \
  --min-accuracy 0.76 --min-syntax-rate 0.85 --no-require-delimiters \
  --eval-sample-size 50 --eval-min-examples-before-threshold-stop 50 \
  --eval-max-steps 200 --eval-step-token-budget 1 --eval-max-seconds-per-example 450 \
  --restart-after-stuck-iters 0 \
  --vllm-max-model-len 16384 \
  --vllm-gpu-memory-utilization 0.45 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --spider-split-file "$SPLIT" --spider-split-name train \
  --output-name "$NAME" \
  > "$LOG" 2>&1
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/iter50_tok0_h8_7b_driver_${STAMP}.log
