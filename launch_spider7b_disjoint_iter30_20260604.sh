#!/bin/bash
# Spider-7B-Instruct disjoint-split retry, 2026-06-04.
# Prior iter-30 run (ralph_7B_spider_disjoint_iter30_20260603) best: 55%/99%
# on train split — below the 0.64 accuracy bar (external IterGen bar: 62%).
# This run deploys the _delimiter_miss_hint fix to give the author cleaner
# feedback signals. Fresh start (no --initial-strategy-file) to explore
# different strategy space after 30 iterations were stuck at 55%.
# Runs on GPU 1 (freed after GSM-7B a21 held-out reeval completes).
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_7B_spider_disjoint_iter30_20260604
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_100x100_seed123.json
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 30 \
  --output-name ralph_7B_spider_disjoint_iter30_20260604 \
  --min-accuracy 0.64 --min-syntax-rate 0.93 \
  --eval-sample-size 100 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 100 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.50 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  > /tmp/ralph_7B_spider_disjoint_iter30_20260604.log 2>&1 &
echo "Spider-7B-DISJOINT-iter30-20260604 pid: $!  -> $OUT  log=/tmp/ralph_7B_spider_disjoint_iter30_20260604.log"
