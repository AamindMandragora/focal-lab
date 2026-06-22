#!/bin/bash
# Spider-1.5B 300x300 seed334 continued synthesis.
# Prior run hit 47%/97% on 100-sample train draw but only 39%/98% on 300 held-out (1pp below bar).
# This run starts from that warmstart strategy and aims for 52%/92% on train draws to push
# held-out above the 40% IterGen bar.
# Runs on GPU 1 (26GB free after GSM-1.5B reeval exited).
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/spider1p5b_300x300_seed334_cont_20260604
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
STRATEGY=/home/aadivyar/csd-generation/spider1p5b_300x300_warmstart_body.dfy
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 30 \
  --initial-strategy-file "$STRATEGY" \
  --output-name spider1p5b_300x300_seed334_cont_20260604 \
  --min-accuracy 0.52 --min-syntax-rate 0.92 \
  --eval-sample-size 100 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 100 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  > /tmp/spider1p5b_300x300_seed334_cont_20260604.log 2>&1 &
echo "Spider-1.5B 300x300 cont pid: $!  -> $OUT  log=/tmp/spider1p5b_300x300_seed334_cont_20260604.log"
