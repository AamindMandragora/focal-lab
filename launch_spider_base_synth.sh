#!/usr/bin/env bash
# Spider-1.5B synthesis targeting the BASE Qwen2.5-Coder-1.5B in completion mode.
# Goal: beat IterGen base bar (63.7% acc / 88.7% syntax) on the held-out 100.
set -euo pipefail
cd /home/aadivyar/csd-generation

RUN=spider_base_coder1p5b_completion_600steps_20260531
LOGDIR="logs/${RUN}"
mkdir -p "$LOGDIR"

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

nohup python synthesis/run_synthesis.py \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-backend bedrock \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --anthropic-effort high \
  --eval-model Qwen/Qwen2.5-Coder-1.5B \
  --eval-backend vllm \
  --completion-mode \
  --spider-split-file environment/benchmark_splits/spider_dev_proportional.json \
  --spider-split-name train \
  --eval-sample-size 50 \
  --eval-max-steps 600 \
  --eval-min-examples-before-threshold-stop 50 \
  --min-accuracy 0.64 \
  --min-syntax-rate 0.89 \
  --require-delimiters \
  --helper-selection-policy bandit \
  --max-iterations 15 \
  -o "$RUN" \
  > "${LOGDIR}/run.log" 2>&1 &

echo "PID=$!"
echo "LOG=${LOGDIR}/run.log"
