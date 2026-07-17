#!/usr/bin/env bash
# Focused re-run: 1.5B CRANE (CoT+<<>>) ONLY, at --eval-max-seconds-per-example 450.
# The 16384 attempt crashed on context overflow; the 32768 attempt then EARLY-STOPPED
# at 171/300 because CoT on the weak 1.5B is slow (slowest example 344.7s > the 180s
# budget). 450s covers the observed worst case with margin -> clean full N=300. The 7B
# CRANE already finished clean (56.7%/61.0%), so it is NOT re-run here.
# Pure re-eval (max-iterations 1, bars 0, --initial-strategy-file), seed334 test-300,
# official grader, local vLLM author NEVER called -> ZERO Bedrock spend. GPU 2 (shared).
set -u
cd ~/csd-generation
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export SPIDER_TOKEN0_CONSTRAINED=0   # legacy visible-<<>> path (CRANE uses delimiters)
export SPIDER_CRANE_COT=1            # chain-of-thought prompt for the reasoning baseline
export CUDA_VISIBLE_DEVICES=2

STRAT=outputs/strategies/spider_crane_faithful.dfybody
SPLIT=environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider1p5b_crane_cot_n300_seed334_retry450
LOG=logs/${NAME}_${STAMP}.log

echo "[launch] $NAME -> $LOG" | tee -a logs/crane_retry450_driver_${STAMP}.log
python -m synthesis.run_synthesis \
  --task "$TASK" --dataset spider \
  --generation-model Qwen/Qwen2.5-1.5B-Instruct --generation-backend vllm --allow-small-author-model \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 1 --initial-strategy-file "$STRAT" \
  --min-accuracy 0.0 --min-syntax-rate 0.0 \
  --eval-sample-size 300 --eval-min-examples-before-threshold-stop 300 \
  --eval-max-steps 1200 --eval-step-token-budget 1 --eval-max-seconds-per-example 450 \
  --restart-after-stuck-iters 0 \
  --vllm-max-model-len 32768 \
  --vllm-gpu-memory-utilization 0.25 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --spider-split-file "$SPLIT" --spider-split-name test \
  --output-name "$NAME" \
  > "$LOG" 2>&1
echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/crane_retry450_driver_${STAMP}.log
echo "[ALL DONE] 1.5B CRANE retry450 complete at $(date -u)" | tee -a logs/crane_retry450_driver_${STAMP}.log
