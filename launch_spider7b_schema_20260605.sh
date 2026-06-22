#!/bin/bash
# Spider-7B 300x300 seed334 — schema-grammar synthesis run.
# Per-example schema grammar is now active (eval_logic.py build_dynamic_parser
# returns a schema-restricted parser). Warmstart from the 63% strategy.
# Need 68%+ training accuracy to reach 62%+ held-out (5-6pp generalization gap).
set -u
cd /home/aadivyar/csd-generation

SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
STRATEGY=/home/aadivyar/csd-generation/spider7b_300x300_a2_63pct_body.dfy
OUT_NAME=spider7b_seed334_schema_grammar_20260605

CUDA_VISIBLE_DEVICES=2 \
LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} \
nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 40 \
  --initial-strategy-file "$STRATEGY" \
  --output-name "$OUT_NAME" \
  --min-accuracy 0.68 --min-syntax-rate 0.92 \
  --eval-sample-size 100 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 100 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 5 \
  --vllm-gpu-memory-utilization 0.50 --device auto \
  --output-dir outputs/generated/"$OUT_NAME" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  > /tmp/${OUT_NAME}.log 2>&1 &
echo "Spider-7B schema grammar pid: $!  -> outputs/generated/$OUT_NAME  log=/tmp/${OUT_NAME}.log"
