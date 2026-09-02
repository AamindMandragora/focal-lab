#!/bin/bash
# H1 test: held-out reeval for GSM-1.5B seed429 warmstart strategy AFTER adding the
# force-close-at-last-complete-prefix fallback (RollbackConstrainedToComplete) to the
# library + strategy. Baseline before this edit: 28/49 acc (57.1%), 43/49 syn (87.8%).
# Goal: syntax >= 90% (>=45/49), accuracy > 32% (>=16/49). Same split/flags as the
# baseline reeval; only the strategy/library edit, output name, and GPU differ.
set -eu

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

cd /home/aadivyar/csd-generation

STRATEGY=/home/aadivyar/csd-generation/gsm1p5b_seed429_warmstart_body.dfy
echo "Strategy: $STRATEGY"

GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed429.json
OUT=outputs/generated/gsm1p5b_seed429_warmstart_heldout_forceclose_20260608
mkdir -p "$OUT"

/apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
    --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
    --dataset gsm_symbolic \
    --max-iterations 1 \
    --initial-strategy-file "$STRATEGY" \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --eval-backend vllm \
    --output-name "gsm1p5b_seed429_warmstart_heldout_forceclose_20260608" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 49 \
    --eval-max-steps 900 \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 90 \
    --eval-min-examples-before-threshold-stop 49 \
    --vllm-gpu-memory-utilization 0.40 \
    --device auto \
    --output-dir "$OUT" \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --anthropic-thinking enabled \
    --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --vllm-tensor-parallel-size 1 \
    --gsm-split-file "$GSM_SPLIT" \
    --gsm-split-name eval \
    2>&1 | tee /tmp/gsm1p5b_seed429_warmstart_heldout_forceclose_20260608.log
