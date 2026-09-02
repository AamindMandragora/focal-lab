#!/bin/bash
# H3: force the synthesis loop to iterate so the author actually exercises RollbackAndContinue.
# EXACTLY ONE change vs the H2 launch (launch_gsm1p5b_rollbackcontinue_20260608.sh):
#   --min-syntax-rate 0.85  ->  0.92   (above the warmstart's 87.8% train syntax, so the
#                                       seed no longer clears the bar at attempt 1)
# Everything else identical: same deployed tool, same clean warmstart baseline body,
# --min-accuracy 0.31, --max-iterations 15, Sonnet-4-6 author, seed429 TRAIN split.
# The _span_not_closed_hint feedback edit is HELD (not deployed) to keep this one variable.
set -uo pipefail
cd /home/aadivyar/csd-generation

export CUDA_VISIBLE_DEVICES=1,2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

OUT=outputs/generated/ralph_1p5B_gsm_rollbackcontinue_h3_20260608
GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed429.json
LOG=/tmp/ralph_1p5B_gsm_rollbackcontinue_h3_20260608.log

mkdir -p "$OUT"

nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_1p5B_gsm_rollbackcontinue_h3_20260608 \
  --output-dir "$OUT" \
  --min-accuracy 0.31 --min-syntax-rate 0.92 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 120 --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.18 --device auto \
  --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --gsm-split-file "$GSM_SPLIT" --gsm-split-name train \
  --initial-strategy-file /home/aadivyar/csd-generation/gsm1p5b_seed429_warmstart_body.dfy \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
echo "OUT=$OUT"
