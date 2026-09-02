#!/bin/bash
# Held-out reeval for Spider-1.5B attempt 14 strategy (41%/98% on train split).
# Evaluates on the held-out 100-example eval split.
# a14: OpenConstrainedSpan + ConfidenceGatedStep, verified by Dafny.
set -e

export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

cd /home/aadivyar/csd-generation

python -m synthesis.run_synthesis \
    --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
    --dataset spider \
    --max-iterations 1 \
    --initial-strategy-file /home/aadivyar/csd-generation/a14_spider1p5b_strategy.dfy \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --eval-backend vllm \
    --output-name "spider1p5b_a14_heldout_reeval_20260604" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 100 \
    --eval-max-steps 1200 \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 180 \
    --eval-min-examples-before-threshold-stop 100 \
    --vllm-gpu-memory-utilization 0.20 \
    --device auto \
    --output-dir outputs/generated/spider1p5b_a14_heldout_reeval_20260604 \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --anthropic-thinking enabled \
    --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --vllm-tensor-parallel-size 1 \
    --spider-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_100x100_seed123.json \
    --spider-split-name eval \
    2>&1 | tee /tmp/spider1p5b_a14_heldout_reeval_20260604.log
