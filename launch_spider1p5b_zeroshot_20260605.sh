#!/bin/bash
# Spider-1.5B 300x300 seed334 held-out reeval WITH zero-shot IterGen-aligned prompt.
# The zero-shot prompt change is in the shared eval path so it affects 1.5B too;
# this relaunch both applies the new prompt and guards the existing 44%/97% win.
# Same warmstart winner strategy, same seed334 eval-side 300 (== test side).
# Runs on GPU 1.
set -eu

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

cd /home/aadivyar/csd-generation

STRATEGY=/home/aadivyar/csd-generation/spider1p5b_300x300_warmstart_body.dfy
echo "Strategy: $STRATEGY"

SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
OUT=outputs/generated/spider1p5b_300x300_seed334_heldout_zeroshot_20260605
mkdir -p "$OUT"

nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
    --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
    --dataset spider \
    --max-iterations 1 \
    --initial-strategy-file "$STRATEGY" \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --eval-backend vllm \
    --output-name "spider1p5b_300x300_seed334_heldout_zeroshot_20260605" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 300 \
    --eval-max-steps 1200 \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 180 \
    --eval-min-examples-before-threshold-stop 300 \
    --vllm-gpu-memory-utilization 0.20 \
    --device auto \
    --output-dir "$OUT" \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --anthropic-thinking enabled \
    --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --vllm-tensor-parallel-size 1 \
    --spider-split-file "$SPIDER_SPLIT" \
    --spider-split-name eval \
    > /tmp/spider1p5b_300x300_seed334_heldout_zeroshot_20260605.log 2>&1 &
echo "Spider-1.5B zero-shot held-out eval pid: $!  -> $OUT  log=/tmp/spider1p5b_300x300_seed334_heldout_zeroshot_20260605.log"
