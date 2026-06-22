#!/bin/bash
# Held-out reeval for GSM-1.5B seed429 warmstart run (42.86%/87.75% on train split).
# Evaluates on the held-out 49-example eval side of the seed429 split.
# Runs on GPU 1 (mostly free).
set -eu

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

cd /home/aadivyar/csd-generation

STRATEGY=/home/aadivyar/csd-generation/gsm1p5b_seed429_warmstart_body.dfy
echo "Strategy: $STRATEGY"

GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed429.json
OUT=outputs/generated/gsm1p5b_seed429_warmstart_inertness_reeval_20260608
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
    --output-name "gsm1p5b_seed429_warmstart_inertness_reeval_20260608" \
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
    2>&1 | tee /tmp/gsm1p5b_seed429_warmstart_inertness_reeval_20260608.log
echo "GSM-1.5B seed429 held-out reeval done -> $OUT"
