#!/bin/bash
# Spider-1.5B relaunch 2026-05-29 on GPU 2 (after the crane_faithful GSM baseline
# finished there). Mirrors launch_spider1p5b_maxsteps1200.sh EXACTLY except GPU
# index (3->2), output dir/name, and log path. Keeps util at 0.17 (the proven
# template value) because GPU 2 still has a ~16.7GB co-tenant (only ~24GB free);
# 0.17 (~6.8GB) is a safe neighbor and util only sizes the KV cache.
# Keeps the doubled step budget (1200) + 180s wall cap so a forced/natural `>>`
# close has room to fire, the libstdc++ CXXABI LD_LIBRARY_PATH fix, and the
# backtick-delimited task string verbatim.
# OPERATIONAL ONLY: no experimental parameter, prompt, or code changes.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_spider_relaunch_20260529
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 10 \
  --output-name ralph_1p5B_spider_relaunch_20260529 \
  --min-accuracy 0.52 --min-syntax-rate 0.85 \
  --eval-sample-size 50 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.17 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  > /tmp/ralph_1p5B_spider_relaunch_20260529.log 2>&1 &
echo "Spider-1.5B pid: $!  -> $OUT  log=/tmp/ralph_1p5B_spider_relaunch_20260529.log"
