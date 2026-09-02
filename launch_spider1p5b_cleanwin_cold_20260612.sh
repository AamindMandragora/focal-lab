#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START (2026-06-12).
# Goal: beat in-house IterGen (52.0%/94.7 held-out) cleanly, not by tie.
# Bar raised to 0.57 acc / 0.92 syn on N=100 train draws (vs rebar52 bar of 0.56).
# Key differences from rebar52 (launch_spider1p5b_rebar52_20260611.sh):
#   - NO --initial-strategy-file (cold start — warm anchoring proven trap: rebar52 only micro-varied ConfidenceGatedStep)
#   - --min-accuracy 0.57 (rebar52 used 0.56; higher bar for a clean gap over 52.0 held-out)
#   - Author sees the new RegenerateUnitOnCheckFailure helper via --no-adaptive-helper-mask
#     (deploy_itergen_helper.sh must have landed before this launches)
# Preserved exactly from rebar52: task text, env setup, SPIDER_DB_DIR, CSD_API_MAX_RETRIES,
#   timeout flags, beam flags, vllm-tensor-parallel-size, --device auto.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
OUT=outputs/generated/spider1p5b_cleanwin_cold_20260612
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_cleanwin_cold_20260612 \
  --min-accuracy 0.57 --min-syntax-rate 0.92 \
  --eval-sample-size 100 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 300 --eval-min-examples-before-threshold-stop 100 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --no-adaptive-helper-mask --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train
echo "SYNTH_EXIT=$?"
echo "DONE_SPIDER1P5B_CLEANWIN $(date)"
