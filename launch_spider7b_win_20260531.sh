#!/bin/bash
# Spider-7B-Instruct WIN run 2026-05-31 on GPU 2 (fully free, 40GB).
# Target: beat the in-house apples-to-apples IterGen bar of 62% acc / 92% syntax
# (N=100, native itergen harness, greedy, schema-aware backtracking). To win we
# need acc AND syntax strictly above that, so thresholds are set just above the bar
# (0.64 / 0.93). Mirrors launch_spider1p5b_relaunch_20260529.sh EXACTLY except GPU
# index, eval-model (1.5B->7B), util (0.17->0.50 for the larger weights), output
# dir/name, thresholds, and log path. No experimental prompt/code changes.
# Sonnet-4.6 authors the strategy (no strategy guidance in prompts).
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_7B_spider_win_20260531
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 10 \
  --output-name ralph_7B_spider_win_20260531 \
  --min-accuracy 0.64 --min-syntax-rate 0.93 \
  --eval-sample-size 50 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.50 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  > /tmp/ralph_7B_spider_win_20260531.log 2>&1 &
echo "Spider-7B-WIN pid: $!  -> $OUT  log=/tmp/ralph_7B_spider_win_20260531.log"
