#!/bin/bash
# Spider-1.5B re-synthesis vs the NEW in-house IterGen bar (2026-06-11).
# Trigger: in-house IterGen head-to-head on the exact seed334 test-300 scored 52.0%/94.7,
# beating our recorded 51.0% by 3 examples — user rule: baseline beats us -> re-run synthesis
# against the higher bar. Warm from the current best body (spider1p5b_300x300_warmstart_body.dfy,
# the 51.0% strategy under the DB-fixed scorer).
# Lessons applied vs the 06-04 template:
#   - --no-adaptive-helper-mask (bandit mask hid key helpers from the author; 06-11 finding)
#   - SPIDER_DB_DIR pinned to the test-suite databases (scorer == official grader)
#   - CSD_API_MAX_RETRIES=10 (survive Bedrock daily-quota windows)
#   - train bar 0.56/0.92 (margin over the 52.0 held-out target; in-loop scorer now matches official)
# Task text is the original bare format contract — clean under the no-mechanism-guidance rule.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
OUT=outputs/generated/spider1p5b_seed334_rebar52_20260611
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
STRATEGY=/home/aadivyar/csd-generation/spider1p5b_300x300_warmstart_body.dfy

CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --initial-strategy-file "$STRATEGY" \
  --output-name spider1p5b_seed334_rebar52_20260611 \
  --min-accuracy 0.56 --min-syntax-rate 0.92 \
  --eval-sample-size 100 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 100 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --no-adaptive-helper-mask --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train
echo "EXIT_SYNTH_SPIDER1P5B_REBAR52=$?"
echo "DONE_SPIDER1P5B_REBAR52 $(date)"
