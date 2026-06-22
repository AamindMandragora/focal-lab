#!/bin/bash
# Spider-7B 300x300 seed334 — restart with 1.5B warmstart body.
# Prior run (PID 1086796) got stuck after attempt 4 (best: 56%/95%, all attempts had
# 0/100 constraint engagement — reactive triggers never fired, OpenConstrainedSpan fired
# after 5 free steps too late for 7B which generates answer in ~4 tokens).
# This run starts from the 1.5B warmstart which calls OpenConstrainedSpan FIRST,
# guaranteeing constraint entry before any unconstrained generation.
# Runs on GPU 3 (free after prior run killed).
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/spider7b_300x300_seed334_warmstart_20260604
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
STRATEGY=/home/aadivyar/csd-generation/spider1p5b_300x300_warmstart_body.dfy
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 30 \
  --initial-strategy-file "$STRATEGY" \
  --output-name spider7b_300x300_seed334_warmstart_20260604 \
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
  > /tmp/spider7b_300x300_seed334_warmstart_20260604.log 2>&1 &
echo "Spider-7B 300x300 warmstart pid: $!  -> $OUT  log=/tmp/spider7b_300x300_seed334_warmstart_20260604.log"
