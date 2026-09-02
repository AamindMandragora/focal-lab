#!/bin/bash
# Held-out reeval for GSM-7B a21 strategy (75.5%/91.8% on train split)
# Evaluates on the held-out 49-example eval split to get reportable numbers.
# a21: per-span-budget-40 + rollback strategy, verified by Dafny.
set -e

export CUDA_VISIBLE_DEVICES=1,2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH}

cd /home/aadivyar/csd-generation

python -m synthesis.run_synthesis \
    --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters." \
    --dataset gsm_symbolic \
    --max-iterations 1 \
    --initial-strategy-file /home/aadivyar/csd-generation/a21_gsm7b_strategy.dfy \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "Qwen/Qwen2.5-7B-Instruct" \
    --eval-backend vllm \
    --output-name "gsm7b_a21_heldout_reeval_20260604" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 49 \
    --eval-max-steps 900 \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 90 \
    --eval-min-examples-before-threshold-stop 49 \
    --vllm-gpu-memory-utilization 0.45 \
    --device auto \
    --output-dir outputs/generated/gsm7b_a21_heldout_reeval_20260604 \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --anthropic-thinking enabled \
    --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --vllm-tensor-parallel-size 1 \
    --gsm-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json \
    --gsm-split-name eval \
    2>&1 | tee /tmp/gsm7b_a21_heldout_reeval_20260604.log
