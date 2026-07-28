#!/usr/bin/env bash
# H7 token-0 example run — 1.5B, 50-example seed334-train subset. IDENTICAL fair recipe to
# run_iter50_cold_1p5b.sh; the ONLY scientific change is the new verified "// Token-0 grounded
# constrained CSD." example now in prompts.py VERIFIED_EXAMPLES. GPU moved 2->1 because my own
# orphaned procs (not killable: shared-server safety guard) hold GPU 2/1 memory; GPU 1 has 16GB
# free, enough for the 1.5B at util 0.30 (~12GB). All result-affecting flags unchanged.
# Author = UIUC focal lab AWS account 887730490125 (AWS_BEARER_TOKEN_BEDROCK, us-east-1), user-approved.
set -u
cd ~/csd-generation
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=1
SPLIT=environment/benchmark_splits/spider_dev_proportional_50train_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider1p5b_iter50_tok0_cold_${STAMP}
LOG=logs/${NAME}.log
echo "[launch] $NAME -> $LOG (author=bedrock sonnet-4.6, COLD, token-0, 50-set train, bar 0.66, H7 example)" | tee -a logs/iter50_tok0_driver_${STAMP}.log
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
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/iter50_tok0_driver_${STAMP}.log
