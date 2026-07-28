#!/usr/bin/env bash
# Cycle-0 CRANE baseline: reasoning-based CRANE = chain-of-thought prompt
# (SPIDER_CRANE_COT=1) + visible << >> constrained spans (legacy path,
# SPIDER_TOKEN0_CONSTRAINED=0), strategy body = helpers.CraneGeneration(...).
# Pure re-eval (max-iterations 1, bars 0, --initial-strategy-file) on the exact
# seed334 test-300 split, official grader, full N=300, for BOTH non-Coder eval
# models. Author backend is local vLLM and is NEVER called (feedback_loop.py:1647
# uses the provided body directly; generator loads lazily) -> ZERO Bedrock spend.
# Sequential on GPU 2 (shared with another user). Run AFTER the GCD-floor baseline.
#
# 2026-06-22 RE-RUN: --vllm-max-model-len 32768 (Qwen2.5 native context). The
# first attempt at the 16384 default CRASHED: CRANE's free CoT reasoning grew the
# running context past 16384 on a large-schema example, vLLM raised a fatal
# VLLMValidationError and killed the engine core (whole run aborted, exit 1, no
# report). 32768 gives the reasoning room to fit. Utils bumped (1.5B 0.25, 7B 0.50)
# for the doubled KV cache on the shared card.
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

run_one () {
  local EVAL_MODEL="$1"; local UTIL="$2"; local NAME="$3"
  local LOG=logs/${NAME}_${STAMP}.log
  echo "[launch] $NAME  eval=$EVAL_MODEL util=$UTIL  -> $LOG" | tee -a logs/crane_baseline_driver_${STAMP}.log
  python -m synthesis.run_synthesis \
    --task "$TASK" --dataset spider \
    --generation-model Qwen/Qwen2.5-1.5B-Instruct --generation-backend vllm --allow-small-author-model \
    --eval-model "$EVAL_MODEL" --eval-backend vllm \
    --max-iterations 1 --initial-strategy-file "$STRAT" \
    --min-accuracy 0.0 --min-syntax-rate 0.0 \
    --eval-sample-size 300 --eval-min-examples-before-threshold-stop 300 \
    --eval-max-steps 1200 --eval-step-token-budget 1 --eval-max-seconds-per-example 180 \
    --restart-after-stuck-iters 0 \
    --vllm-max-model-len 32768 \
    --vllm-gpu-memory-utilization "$UTIL" --device auto --vllm-tensor-parallel-size 1 \
    --adaptive-helper-mask --helper-selection-policy bandit \
    --spider-split-file "$SPLIT" --spider-split-name test \
    --output-name "$NAME" \
    > "$LOG" 2>&1
  echo "[done] $NAME exit=$? at $(date -u)" | tee -a logs/crane_baseline_driver_${STAMP}.log
}

run_one "Qwen/Qwen2.5-1.5B-Instruct" 0.25 "spider1p5b_crane_cot_n300_seed334"
run_one "Qwen/Qwen2.5-7B-Instruct"   0.50 "spider7b_crane_cot_n300_seed334"
echo "[ALL DONE] CRANE CoT+<<>> baselines complete at $(date -u)" | tee -a logs/crane_baseline_driver_${STAMP}.log
