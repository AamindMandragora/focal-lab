#!/usr/bin/env bash
# PURE RE-EVAL (no synthesis) of the bar057-run 1.5B ACCEPTED strategy on the HELD-OUT 300-TEST set (seed334).
# Stage 3 of the promote discipline: iterate(300train) -> final(300test).
#
# Strategy provenance: success_report.strategy_code of run
#   spider1p5b_300train_cold_iter40_bar057_20260625_075409  (ACCEPTED "SUCCESS after 29 attempt(s)";
#   181/300 = 60.33% acc, 98.33% syntax, N=300 TRAIN, COLD, mask ON + bandit, token-0,
#   --no-require-delimiters). It crossed the 0.57 accept bar. Body copied verbatim to
#   win_1p5b_bar057_300train_body.dfy (md5 ec44493b074aceb063f05dd249f23b34).
#
# This is the TRUE number vs in-house IterGen 1.5B = 52.0% (156/300, ~94.7% syn) on the SAME seed334
# test-300. Author iterated only on train_indices; test_indices are disjoint (overlap 0) and the
# Sonnet author never saw them. Eval recipe IDENTICAL to the training run so train->test is fair.
#
# $0 Bedrock: --max-iterations 1 + --initial-strategy-file => evaluated ONCE on 300 examples; the
# Sonnet-4.6 author is never called. Bars 0 (pure measurement) + min-examples 300 forces full 300.
# GPU: export GPU=<n>; defaults to 0.
set -u
cd ~/csd-generation
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=${GPU:-0}
SPLIT=environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
BODY=saved-results/spider-goal-strategies/win_1p5b_bar057_300train_body.dfy
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider1p5b_bar057win_reeval_300TEST_seed334_${STAMP}
LOG=logs/${NAME}.log
echo "[launch] $NAME -> $LOG (PURE RE-EVAL of 1.5B bar057 accepted strategy on 300-TEST, max-iter 1, \$0 Bedrock, GPU=$CUDA_VISIBLE_DEVICES)" | tee -a logs/${NAME}_driver.log
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
  --spider-split-file "$SPLIT" --spider-split-name test \
  --output-name "$NAME" \
  > "$LOG" 2>&1
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/${NAME}_driver.log
