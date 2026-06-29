#!/bin/bash
# GSM-1.5B relaunch 2026-05-30 — FIXED-LIBRARY run.
# Identical synthesis config to launch_gsm1p5b_relaunch_20260529_detector.sh.
# The ONE change being tested: synthesis/verify/library/VerifiedAgentSynthesis.dfy now
# closes a constrained span with the RENDERED-TEXT-SUFFIX dedup (RenderedEndsWith) instead
# of the old exact-token guard. That fixes the doubled-'>>' exit bug: when the model emits
# the closing delimiter in any surface form other than a single ">>" token (space-prefixed
# " >>", or split ">"+">"), the old guard failed to dedup and appended a SECOND ">>",
# producing <<expr>>>> which the grammar rejected. att14's best strategy capped at 0.76
# syntax for exactly this reason.
#
# Targeted-experiment evidence (att14's exact compiled strategy on the fixed library,
# same 50 GSM examples / same eval config): syntax 0.76 -> 0.90 (7/8 doubled-'>>' cases now
# close with one ">>"), accuracy held 0.56 -> 0.52 (within N=50 noise; inner math unchanged).
# Purpose of THIS run: with the exit bug gone, let the author evolve a strategy that clears
# BOTH the 0.32 accuracy bar and the 0.90 syntax bar (the win), instead of being early-
# stopped as "threshold-impossible" by a library bug.
# New output dir/name/log; GPU 2 (least loaded). Nothing else changed.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_gsm_relaunch_20260530_fixedlib
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_1p5B_gsm_relaunch_20260530_fixedlib \
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
  > /tmp/ralph_1p5B_gsm_relaunch_20260530_fixedlib.log 2>&1 &
echo "GSM-1.5B fixedlib pid: $!  -> $OUT  log=/tmp/ralph_1p5B_gsm_relaunch_20260530_fixedlib.log"
