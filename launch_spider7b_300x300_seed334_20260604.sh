#!/bin/bash
# Spider-7B synthesis on 300x300 seed334 split, 2026-06-04.
# Advisor requires 300-train + 300-held-out for paper comparison.
# Fresh start (no warmstart) — prior 100x100 strategies overfit (-13pp held-out drop).
# --eval-sample-size 100 draws 100 random examples per iteration from the
# 300-example training pool (no --eval-seed => fresh random draw each time).
# Threshold raised to 0.67/0.95 (vs 0.62/0.92 bar) as buffer for N=100 noise.
# Runs on GPU 1.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_7B_spider_300x300_seed334_20260604
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 30 \
  --output-name ralph_7B_spider_300x300_seed334_20260604 \
  --min-accuracy 0.67 --min-syntax-rate 0.95 \
  --eval-sample-size 100 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 100 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.50 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  > /tmp/ralph_7B_spider_300x300_seed334_20260604.log 2>&1 &
echo "Spider-7B-300x300 pid: $!  -> $OUT  log=/tmp/ralph_7B_spider_300x300_seed334_20260604.log"
