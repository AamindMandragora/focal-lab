#!/bin/bash
# Spider-1.5B synthesis on 300x300 seed334 split, 2026-06-04.
# Advisor requires 300-train + 300-held-out for paper comparison.
# Warmstart from a14 strategy (OpenConstrainedSpan+ConfidenceGatedStep),
# the same body that got 44%/97% on the seed334 100-example held-out.
# --eval-sample-size 100 draws 100 random examples per iteration from the
# 300-example training pool (no --eval-seed => fresh random draw each time).
# Threshold raised to 0.46/0.90 (vs 0.40/0.86 bar) as buffer for N=100 noise.
# Runs on GPU 2.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_spider_300x300_seed334_20260604
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
WARMSTART=/home/aadivyar/csd-generation/a14_spider1p5b_300x300_warmstart_body.dfy
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 30 \
  --output-name ralph_1p5B_spider_300x300_seed334_20260604 \
  --min-accuracy 0.46 --min-syntax-rate 0.90 \
  --eval-sample-size 100 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 100 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  --initial-strategy-file "$WARMSTART" \
  > /tmp/ralph_1p5B_spider_300x300_seed334_20260604.log 2>&1 &
echo "Spider-1.5B-300x300 pid: $!  -> $OUT  log=/tmp/ralph_1p5B_spider_300x300_seed334_20260604.log"
