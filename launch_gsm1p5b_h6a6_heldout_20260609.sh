#!/bin/bash
# Held-out validation of the H6 anchor strategy (attempt 6: 89.8% train syntax / 42.9%
# train acc, 5 unclosed spans). Measures the REAL goal metric on the held-out split.
#
# This is a MEASUREMENT, not an experiment: --max-iterations 1 evaluates the provided
# initial strategy ONCE on the seed123 EVAL split and never refines it. Apples-to-apples
# with how the baseline (57.1% acc / 87.8% syntax) was measured -- identical eval params,
# same eval model, same split file, same sample size. ONLY differences vs that baseline
# measurement: the strategy body (attempt 6 instead of warmstart) and a free GPU.
#
# FIXES vs the prior failed side-agent launch (PID 3493282, which crashed at argparse):
#   1. ADDS the REQUIRED flags --min-accuracy / --min-syntax-rate (their omission was the
#      crash cause). At --max-iterations 1 these are inert for the eval numbers -- they
#      only gate loop early-stop, and there is exactly one iteration.
#   2. Uses the VERIFIED-CLEAN strategy file gsm1p5b_h6_attempt6_anchor_body.dfy
#      (subagent-checked: starts // CSD_RATIONALE_BEGIN, ends cost := steps;, no log
#      noise) instead of the prior run's unverified /tmp/h6_attempt6_body.dfy.
#   3. CUDA_VISIBLE_DEVICES=0 (cleanly free GPU; avoids any lingering H6 state on 1/2).
set -uo pipefail
cd /home/aadivyar/csd-generation

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

OUT=outputs/generated/ralph_1p5B_gsm_h6a6_heldout_20260609
EVAL_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
STRAT=/home/aadivyar/csd-generation/gsm1p5b_h6_attempt6_precise_body.dfy
LOG=/tmp/ralph_1p5B_gsm_h6a6_heldout_20260609.log

mkdir -p "$OUT"

nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 1 \
  --output-name ralph_1p5B_gsm_h6a6_heldout_20260609 \
  --output-dir "$OUT" \
  --min-accuracy 0.31 --min-syntax-rate 0.92 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 120 --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.18 --device auto \
  --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --gsm-split-file "$EVAL_SPLIT" --gsm-split-name eval \
  --initial-strategy-file "$STRAT" \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
echo "OUT=$OUT"
