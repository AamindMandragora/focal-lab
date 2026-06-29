#!/bin/bash
# Spider-7B clean-win campaign — COLD START (2026-06-12).
# Goal: beat in-house IterGen (65.7%/? held-out) cleanly — currently a 1-example tie (65.3).
# Bar: 0.68 acc / 0.93 syn on N=100 train draws.
# Rationale: a2 history shows +2.3pp train→held-out generalization (train 63 → held-out 65.3),
#   so 68% train should translate to ≥198/300 = 66.0% held-out, beating IterGen's 65.7.
# Key differences from rebar52 base (launch_spider1p5b_rebar52_20260611.sh):
#   - NO --initial-strategy-file (cold start — same warm-anchoring trap applies to 7B)
#   - --eval-model Qwen/Qwen2.5-7B-Instruct (vs 1.5B)
#   - --vllm-gpu-memory-utilization 0.50 (vs 0.20; 7B needs ~18GB — wait for 14B sweep placement)
#   - --min-accuracy 0.68, --min-syntax-rate 0.93 (higher bars for 7B cell)
#   - Author sees the new RegenerateUnitOnCheckFailure helper via --no-adaptive-helper-mask
#     (deploy_itergen_helper.sh must have landed before this launches)
# Preserved exactly from rebar52: task text, env setup, SPIDER_DB_DIR, CSD_API_MAX_RETRIES,
#   timeout flags, beam flags, vllm-tensor-parallel-size, --device auto.
# NOTE: launch ONLY after confirming GPU memory (≥18GB free needed at util 0.55).
#   If the 14B sweep is still running, wait for its placement to settle before launching.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
OUT=outputs/generated/spider7b_cleanwin_cold_20260612
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider7b_cleanwin_cold_20260612 \
  --min-accuracy 0.68 --min-syntax-rate 0.93 \
  --eval-sample-size 100 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 300 --eval-min-examples-before-threshold-stop 100 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.50 --device auto \
  --output-dir "$OUT" \
  --no-adaptive-helper-mask --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train
echo "SYNTH_EXIT=$?"
echo "DONE_SPIDER7B_CLEANWIN $(date)"
