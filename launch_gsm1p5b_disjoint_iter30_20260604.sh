#!/bin/bash
# GSM-1.5B-Instruct disjoint-split retry, 2026-06-04.
# Prior iter-30 run (ralph_1p5B_gsm_disjoint_iter30_20260603) best: 51%/87.8%
# on train split — acc clears 0.32 bar but syntax stuck at 87.8% (bar 0.90).
# This run deploys the _delimiter_miss_hint fix (suppresses false "no spans"
# warning when spans ARE being produced) to give the author cleaner signals.
# Fresh start (no --initial-strategy-file) to escape the 87.8% syntax plateau.
# Runs on GPU 3.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_gsm_disjoint_iter30_20260604
mkdir -p "$OUT"
GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 30 \
  --output-name ralph_1p5B_gsm_disjoint_iter30_20260604 \
  --min-accuracy 0.32 --min-syntax-rate 0.90 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.40 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file "$GSM_SPLIT" --gsm-split-name train \
  > /tmp/ralph_1p5B_gsm_disjoint_iter30_20260604.log 2>&1 &
echo "GSM-1.5B-DISJOINT-iter30-20260604 pid: $!  -> $OUT  log=/tmp/ralph_1p5B_gsm_disjoint_iter30_20260604.log"
