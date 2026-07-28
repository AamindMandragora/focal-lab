#!/usr/bin/env bash
# PURE RE-EVAL (no synthesis) of the accepted H8 1.5B winner on the 300-example seed334 TRAIN set.
# Stage 2 of the promote discipline: iterate(50) -> promote(300train) -> final(300test).
#
# Winner provenance: success_report.strategy_code of run
#   spider1p5b_iter50_tok0_h8_iter40_cold_20260624_001438  (38/50 = 76% acc, 100% syntax, N=50,
#   early_stopped=False; fair = AppendTaskGuidance format string is model-derived, att10 CLEAN
#   precedent; COLD, mask ON + bandit). Body copied verbatim to the durable path below.
#
# $0 Bedrock: --max-iterations 1 + --initial-strategy-file => the provided strategy is evaluated
# ONCE on 300 examples; the Sonnet-4.6 author is never called. Author = UIUC focal lab AWS account
# 887730490125 (AWS_BEARER_TOKEN_BEDROCK, us-east-1), user-approved campaign spend (here: zero).
#
# Eval recipe is IDENTICAL to the H8 synthesis run so the 50->300 comparison is fair: token-0,
# eval-max-steps 200, step-token-budget 1, 450s/example, mask ON + bandit, no-require-delimiters,
# vllm max-model-len 16384, util 0.30. Bars set to 0 (pure measurement) + min-examples 300 forces
# the full 300 to run (no threshold early-stop) so the reported accuracy is over a true N=300.
set -u
cd ~/csd-generation
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=2
SPLIT=environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
BODY=saved-results/spider-goal-strategies/h8_1p5b_winner_body.dfy
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider1p5b_h8win_reeval_300train_seed334_${STAMP}
LOG=logs/${NAME}.log
echo "[launch] $NAME -> $LOG (PURE RE-EVAL of H8 winner on 300-train, max-iter 1, \$0 Bedrock)" | tee -a logs/${NAME}_driver.log
python -m synthesis.run_synthesis \
  --task "$TASK" --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --initial-strategy-file "$BODY" \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 1 \
  --min-accuracy 0.0 --min-syntax-rate 0.0 --no-require-delimiters \
  --eval-sample-size 300 --eval-min-examples-before-threshold-stop 300 \
  --eval-max-steps 200 --eval-step-token-budget 1 --eval-max-seconds-per-example 450 \
  --restart-after-stuck-iters 0 \
  --vllm-max-model-len 16384 \
  --vllm-gpu-memory-utilization 0.30 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --spider-split-file "$SPLIT" --spider-split-name train \
  --output-name "$NAME" \
  > "$LOG" 2>&1
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/${NAME}_driver.log
