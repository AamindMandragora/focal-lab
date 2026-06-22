#!/bin/bash
# Spider-7B synthesis warm-start from a23 (60%/97% on train, need 62%+ on held-out).
# Best train attempt was 60%, 2pp below 62% bar. Warm-starting from a23 strategy.
# GPU 3 (free after killing GSM-1.5B).
set -u
cd /home/aadivyar/csd-generation
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_100x100_seed334.json
OUT=outputs/generated/ralph_7B_spider_seed334_warmstart_20260604
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} \
nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a SQL query for the given question and database schema, wrapping the SQL inside << >> delimiters.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 30 \
  --output-name ralph_7B_spider_seed334_warmstart_20260604 \
  --min-accuracy 0.62 --min-syntax-rate 0.93 \
  --eval-sample-size 100 --eval-max-steps 600 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.70 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  --initial-strategy-file /home/aadivyar/csd-generation/warmstart_spider7b_a23.dfy \
  > /tmp/ralph_7B_spider_seed334_warmstart_20260604.log 2>&1 &
echo "Spider-7B-warmstart pid: $!  -> $OUT  log=/tmp/ralph_7B_spider_seed334_warmstart_20260604.log"
