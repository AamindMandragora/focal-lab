#!/bin/bash
# Reproducibility verification (2026-06-11): re-run the stored GSM-1.5B winning strategy
# (seed429 warmstart body, held-out 57.1%/87.8% on 06-04, reproduced exactly 06-08) under
# TODAY'S framework. Since the 06-08 inertness reeval, exactly one eval-path file changed
# (evaluator.py grading-timeout guard, 06-10) plus the author-side API retry fix (not in the
# eval path). Grammar/grader/dataset/prompts/helper library unchanged. If this prints
# ~57.1/87.8 again, the win is verified reproducible on the current framework.
# Clone of launch_gsm1p5b_inertness_reeval_20260608.sh — only GPU + output name changed.
set -eu

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

cd /home/aadivyar/csd-generation

STRATEGY=/home/aadivyar/csd-generation/gsm1p5b_seed429_warmstart_body.dfy
echo "Strategy: $STRATEGY"

GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed429.json
OUT=outputs/generated/gsm1p5b_seed429_verify_current_framework_20260611
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
    --output-name "gsm1p5b_seed429_verify_current_framework_20260611" \
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
    2>&1 | tee /tmp/gsm1p5b_seed429_verify_current_framework_20260611.log
echo "DONE_GSM1P5B_VERIFY_20260611"
