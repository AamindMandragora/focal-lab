#!/bin/bash
# Inertness re-eval: Spider-1.5B winner (seed334 eval, expected ~51.0% acc / ~98.0% syn).
# Purpose: verify the global _forbidden_allow_mask fix (blocks {, }, ** tokens) is inert
# for Spider / SQL -- SQL never uses those tokens so no behavior change is expected.
# Strategy: spider1p5b_300x300_warmstart_body.dfy (the 51%/98% winner under DB-fixed scorer)
# Split: spider_dev_proportional_300x300_seed334.json, eval side, N=50 (sanity subset)
# Expected: ~51% acc / ~98% syn (small noise OK; a large drop = regression)
# GPU: 1 only (run AFTER gsm7b inertness reeval finishes; 1.5B needs only 0.15 util)
set -uo pipefail
cd /home/aadivyar/csd-generation

set -a; source /home/aadivyar/csd-generation/.env; set +a

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases

SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
BODY=/home/aadivyar/csd-generation/spider1p5b_300x300_warmstart_body.dfy
OUT=outputs/generated/spider1p5b_inertness_reeval_20260612
mkdir -p "$OUT"

echo "SPIDER1P5B_INERTNESS_START $(date)"
echo "Strategy: $BODY"
echo "Split: $SPIDER_SPLIT (eval side, N=50 subset)"

/apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 1 \
  --initial-strategy-file "$BODY" \
  --output-name spider1p5b_inertness_reeval_20260612 \
  --output-dir "$OUT" \
  --min-accuracy 0.0 --min-syntax-rate 0.0 \
  --eval-sample-size 50 \
  --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 \
  --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.15 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --refinement-beam-size 2 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name eval \
  2>&1 | tee /tmp/spider1p5b_inertness_reeval_20260612.log

EXIT=$?
echo "SPIDER1P5B_INERTNESS_EXIT=$EXIT"
echo "DONE_SPIDER1P5B_INERTNESS $(date)"
