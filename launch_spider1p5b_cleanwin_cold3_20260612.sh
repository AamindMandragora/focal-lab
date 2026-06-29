#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START run #2 (2026-06-12).
# Run #1 ended NO-WIN. Two framework fixes deployed before this launch:
#   FIX A: per-example timeouts now scored as failures; eval continues (not halted).
#          Pathological-strategy guard: stop after 10 timeouts (not 1).
#   FIX B: _unit_rewind_hint in feedback_loop fires when semantic failures dominate
#          and strategy hasn't used RegenerateUnitOnCheckFailure yet.
# Everything else identical to run #1.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
OUT=outputs/generated/spider1p5b_cleanwin_cold3_20260612
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_cleanwin_cold3_20260612 \
  --min-accuracy 0.57 --min-syntax-rate 0.92 \
  --eval-sample-size 300 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 300 --eval-min-examples-before-threshold-stop 300 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --no-adaptive-helper-mask --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train
echo "SYNTH_EXIT=$?"
echo "DONE_SPIDER1P5B_CLEANWIN3 $(date)"
