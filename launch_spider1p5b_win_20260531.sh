#!/bin/bash
# Spider-1.5B-Instruct WIN run 2026-05-31 on GPU 1.
# Target: beat the in-house apples-to-apples IterGen bar of 40% acc / 86% syntax
# (N=100, native itergen harness, greedy, schema-aware backtracking). To win we
# need acc AND syntax strictly above that, so thresholds are set just above the bar
# (0.42 / 0.88). Mirrors launch_spider1p5b_relaunch_20260529.sh EXACTLY except GPU
# index, output dir/name, thresholds, and log path. No experimental prompt/code
# changes. Sonnet-4.6 authors the strategy (no strategy guidance in prompts).
# GPU 1 has ~22GB free (co-tenant ~18GB); util 0.17 (~6.8GB) is a safe neighbor.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_spider_win_20260531
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 10 \
  --output-name ralph_1p5B_spider_win_20260531 \
  --min-accuracy 0.42 --min-syntax-rate 0.88 \
  --eval-sample-size 50 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.17 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  > /tmp/ralph_1p5B_spider_win_20260531.log 2>&1 &
echo "Spider-1.5B-WIN pid: $!  -> $OUT  log=/tmp/ralph_1p5B_spider_win_20260531.log"
