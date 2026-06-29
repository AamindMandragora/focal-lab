#!/bin/bash
# Spider-7B cleanwin cold (att8, 68%/98% on N=100 train) — held-out eval on test split.
# Win condition: accuracy >= 66.0% (198/300) beats IterGen 197/300 (65.7%).
# GPU 1 (~22 GB free, synthesis just exited). LD_LIBRARY_PATH prepended for CXXABI.
set -eu

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

cd /home/aadivyar/csd-generation

STRATEGY=/home/aadivyar/csd-generation/spider7b_cleanwin_cold_20260612_att8_body.dfy
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
OUT=outputs/generated/spider7b_cleanwin_heldout_20260612
LOG=/home/aadivyar/csd-generation/spider7b_cleanwin_heldout_20260612.log

mkdir -p "$OUT"

nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
    --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
    --dataset spider \
    --max-iterations 1 \
    --initial-strategy-file "$STRATEGY" \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --anthropic-thinking enabled \
    --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --eval-model "Qwen/Qwen2.5-7B-Instruct" \
    --eval-backend vllm \
    --output-name "spider7b_cleanwin_heldout_20260612" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 300 \
    --eval-max-steps 1200 \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 300 \
    --eval-min-examples-before-threshold-stop 300 \
    --vllm-gpu-memory-utilization 0.50 \
    --device auto \
    --output-dir "$OUT" \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --vllm-tensor-parallel-size 1 \
    --spider-split-file "$SPIDER_SPLIT" \
    --spider-split-name test \
    > "$LOG" 2>&1 &

PID=$!
echo "Spider-7B cleanwin held-out eval pid=$PID"
echo "  output-dir: $OUT"
echo "  log: $LOG"
echo "  strategy: $STRATEGY"
