#!/bin/bash
# Spider-7B 300x300 seed334 held-out reeval WITH zero-shot IterGen-aligned prompt.
# Same winner strategy (a2 63%/100% train), same seed334 test-side 300, but the
# eval prompt is now zero-shot + terse instruction (keeps << >> wrapper) so the
# only divergence from IterGen is the wrapper. Clean experiment to test whether
# the -8.4pp Spider-7B gap vs IterGen 65.7% is a prompt artifact or method gap.
# Runs on GPU 2.
set -eu

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

cd /home/aadivyar/csd-generation

STRATEGY=/home/aadivyar/csd-generation/spider7b_300x300_a2_63pct_body.dfy
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
OUT=outputs/generated/spider7b_300x300_seed334_heldout_zeroshot_20260605
mkdir -p "$OUT"

nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
    --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
    --dataset spider \
    --max-iterations 1 \
    --initial-strategy-file "$STRATEGY" \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "Qwen/Qwen2.5-7B-Instruct" \
    --eval-backend vllm \
    --output-name "spider7b_300x300_seed334_heldout_zeroshot_20260605" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 300 \
    --eval-max-steps 1200 \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 180 \
    --eval-min-examples-before-threshold-stop 300 \
    --vllm-gpu-memory-utilization 0.50 \
    --device auto \
    --output-dir "$OUT" \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --anthropic-thinking enabled \
    --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --vllm-tensor-parallel-size 1 \
    --spider-split-file "$SPIDER_SPLIT" \
    --spider-split-name test \
    > /tmp/spider7b_300x300_seed334_heldout_zeroshot_20260605.log 2>&1 &
echo "Spider-7B zero-shot held-out eval pid: $!  -> $OUT  log=/tmp/spider7b_300x300_seed334_heldout_zeroshot_20260605.log"
