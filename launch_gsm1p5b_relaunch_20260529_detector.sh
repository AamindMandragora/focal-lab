#!/bin/bash
# GSM-1.5B relaunch 2026-05-29 — DETECTOR run (corrected win bar + constraint-bypass hint).
# Same config as launch_gsm1p5b_relaunch_20260529_syn090.sh (syntax bar 0.90, acc bar
# 0.32, corrected final-block syntax metric). The ONE change being tested: feedback_loop.py
# now emits _constraint_bypassed_hint when << >> delimiters APPEAR in the text but the
# strategy's constrained branch barely engaged (used_constrained_chunk low) — the exact
# GSM-1.5B wall where a reactive `next == "<<"` trigger never matches the space-prefixed
# ' <<' token, so the span content is produced UNCONSTRAINED and syntax caps at ~68%.
# Purpose: verify the new feedback signal pushes the author from a reactive span-entry
# trigger toward FORCING span entry, lifting syntax from ~68% toward the 0.90 bar.
# New output dir/name/log keep the dead syn090 run (no stored results) intact.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_gsm_relaunch_20260529_detector
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_1p5B_gsm_relaunch_20260529_detector \
  --min-accuracy 0.32 --min-syntax-rate 0.90 \
  --eval-sample-size 50 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.40 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json \
  --gsm-split-name eval \
  > /tmp/ralph_1p5B_gsm_relaunch_20260529_detector.log 2>&1 &
echo "GSM-1.5B detector pid: $!  -> $OUT  log=/tmp/ralph_1p5B_gsm_relaunch_20260529_detector.log"
