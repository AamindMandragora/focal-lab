#!/bin/bash
# CHEAP PROBE for Change 1 + grounding + Change 3 (persistent tried-token penalty).
# Purpose (user gate: "probe first, then check in" before the full 20-iter spend):
#   (a) confirm the author USES GenerateWithManagedSpan and VERIFIES on the first try
#       (proof burden removed by Change 1),
#   (b) if the author writes a grounding strategy, confirm the rollback DIVERGES
#       ("[recurrence] penalize" in the eval log; grounding no longer a no-op),
#   (c) no 0000 repetition early-stops.
# COLD start (no --initial-strategy-file). Mask ON. Few attempts + small eval sample
# to bound BOTH Bedrock author cost AND wall-clock. NOT a recorded result.
#
# Author = Bedrock us.anthropic.claude-sonnet-4-6, thinking high (WORK cred
# AWS_BEARER_TOKEN_BEDROCK from project .env, us-east-1 — NOT a personal account).
# GPU 1 at 0.20 mem util (~8GB of 37GB free) — coexists with nalin's small detoxify,
# same GPU the killed r2 used. CSD_RECURRENCE_PENALTY=0.3 (matches IterGen; no-op
# unless a grounding rollback fires).
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3
OUT=outputs/generated/spider1p5b_change3_probe_20260615
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 3 \
  --output-name spider1p5b_change3_probe_20260615 \
  --min-accuracy 0.55 --min-syntax-rate 0.85 \
  --eval-sample-size 40 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 300 --eval-min-examples-before-threshold-stop 40 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train
echo "SYNTH_EXIT=$?"
echo "DONE_SPIDER1P5B_CHANGE3_PROBE $(date)"
