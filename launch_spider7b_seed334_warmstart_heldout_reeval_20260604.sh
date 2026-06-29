#!/bin/bash
# Held-out reeval for Spider-7B seed334 warmstart run (62%/97% on train split).
# Evaluates on the held-out 100-example eval side of the seed334 split.
# Runs on GPU 2 (free).
set -eu

export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

cd /home/aadivyar/csd-generation

STRATEGY=/home/aadivyar/csd-generation/spider7b_seed334_warmstart_body.dfy
echo "Strategy: $STRATEGY"

SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_100x100_seed334.json
OUT=outputs/generated/spider7b_seed334_warmstart_heldout_reeval_20260604
mkdir -p "$OUT"

/apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
    --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
    --dataset spider \
    --max-iterations 1 \
    --initial-strategy-file "$STRATEGY" \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "Qwen/Qwen2.5-7B-Instruct" \
    --eval-backend vllm \
    --output-name "spider7b_seed334_warmstart_heldout_reeval_20260604" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 100 \
    --eval-max-steps 1200 \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 180 \
    --eval-min-examples-before-threshold-stop 100 \
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
    --spider-split-name eval \
    2>&1 | tee /tmp/spider7b_seed334_warmstart_heldout_reeval_20260604.log
echo "Spider-7B seed334 held-out reeval done -> $OUT"
