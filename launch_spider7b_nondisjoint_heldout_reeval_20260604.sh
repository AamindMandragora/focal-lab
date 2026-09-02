#!/bin/bash
# Held-out reeval for Spider-7B non-disjoint winning strategy (68%/98% after regex fix, N=50).
# Tests whether the strategy generalizes to the disjoint held-out eval split (N=100).
# Runs on GPU 2.
set -e

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

cd /home/aadivyar/csd-generation

python -m synthesis.run_synthesis \
    --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
    --dataset spider \
    --max-iterations 1 \
    --initial-strategy-file /home/aadivyar/csd-generation/spider7b_nondisjoint_win_strategy.dfy \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "Qwen/Qwen2.5-7B-Instruct" \
    --eval-backend vllm \
    --output-name "spider7b_nondisjoint_win_heldout_reeval_20260604" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 100 \
    --eval-max-steps 1200 \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 180 \
    --eval-min-examples-before-threshold-stop 100 \
    --vllm-gpu-memory-utilization 0.40 \
    --device auto \
    --output-dir outputs/generated/spider7b_nondisjoint_win_heldout_reeval_20260604 \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --anthropic-thinking enabled \
    --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --vllm-tensor-parallel-size 1 \
    --vllm-max-model-len 4096 \
    --spider-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_100x100_seed123.json \
    --spider-split-name eval \
    2>&1 | tee /tmp/spider7b_nondisjoint_win_heldout_reeval_20260604.log
