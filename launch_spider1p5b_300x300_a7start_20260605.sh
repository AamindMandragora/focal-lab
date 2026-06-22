#!/bin/bash
# Spider-1.5B 300x300 seed334 — fresh synthesis starting from attempt 7 body (49%/84% training).
# Prior cont run (GPU 1, PID 1118413) is stuck oscillating at 48%/97% ceiling; attempt 7 was
# the only attempt to exceed 48% accuracy (49%) but syntax fell to 84%.
# This run starts from attempt 7's body and uses a lower training bar (50%/90%) to allow
# the synthesis to find strategies in the 50%/90%+ range. A 50% training / 90% syntax result
# would give ~41% held-out accuracy and ~90% held-out syntax — enough to satisfy both bars
# (>40% acc, >86% syntax) given the ~9pp training-to-held-out accuracy gap and near-zero
# training-to-held-out syntax gap observed in prior runs.
# Runs on GPU 2 (free after held-out eval PID 1358120 exited).
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/spider1p5b_300x300_a7start_20260605
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
STRATEGY=/home/aadivyar/csd-generation/spider1p5b_a7_49acc_body.dfy
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 30 \
  --initial-strategy-file "$STRATEGY" \
  --output-name spider1p5b_300x300_a7start_20260605 \
  --min-accuracy 0.50 --min-syntax-rate 0.90 \
  --eval-sample-size 100 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 100 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  > /tmp/spider1p5b_300x300_a7start_20260605.log 2>&1 &
echo "Spider-1.5B a7start pid: $!  -> $OUT  log=/tmp/spider1p5b_300x300_a7start_20260605.log"
