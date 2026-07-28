#!/usr/bin/env bash
# Cycle-1 COLD synthesis — 7B, token-0 aligned surface (IterGen-style, NO visible <<>>).
# Identical recipe to run_cycle1_cold_1p5b.sh except eval model = Qwen2.5-7B-Instruct,
# higher train accept bar (--min-accuracy 0.68; the 7B held-out win target is 66.7% and
# held-out typically drops below train), and higher GPU util for the larger eval model.
# Spec: spider-win-goal-spec.md. token-0 default ON (no <<>>), --no-require-delimiters,
# bare task, COLD (no --initial-strategy-file), Bedrock Sonnet-4.6 author, mask ON.
# Synthesis on the TRAIN side of seed334 300x300; win is a later held-out re-eval on TEST.
# SPENDS Bedrock (user approved "both cells" 2026-06-22). GPU 2 (run AFTER 1.5B frees it).
set -u
cd ~/csd-generation
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CUDA_VISIBLE_DEVICES=2

SPLIT=environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider7b_cycle1_token0_cold_${STAMP}
LOG=logs/${NAME}.log

echo "[launch] $NAME -> $LOG  (author=bedrock sonnet-4.6, COLD, token-0, train split)" | tee -a logs/cycle1_cold_driver_${STAMP}.log
python -m synthesis.run_synthesis \
  --task "$TASK" --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --min-accuracy 0.68 --min-syntax-rate 0.85 --no-require-delimiters \
  --eval-sample-size 300 --eval-min-examples-before-threshold-stop 300 \
  --eval-max-steps 1200 --eval-step-token-budget 1 --eval-max-seconds-per-example 300 \
  --restart-after-stuck-iters 0 \
  --vllm-max-model-len 16384 \
  --vllm-gpu-memory-utilization 0.55 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --spider-split-file "$SPLIT" --spider-split-name train \
  --output-name "$NAME" \
  > "$LOG" 2>&1
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/cycle1_cold_driver_${STAMP}.log
echo "[ALL DONE] 7B cycle-1 cold complete at $(date -u)" | tee -a logs/cycle1_cold_driver_${STAMP}.log
