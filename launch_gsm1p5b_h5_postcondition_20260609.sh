#!/bin/bash
# H5: complete RollbackAndContinue's postcondition so the author learns the span is
# left OPEN and closeReserve is the budget reserved FOR the close.
# EXACTLY ONE variable vs the H4 launch: prompts.py:482 now states
#   "The constrained span is left open — no closing delimiter is emitted — and the
#    closeReserve steps withheld from the step budget remain available for closing
#    it afterward."
# That edit is already deployed + verified on focal (FOCAL_PARSE_OK). H4's de-bias
# (line 1900) is KEPT. This launch carries NO new flags vs H4.
# Everything else identical: warmstart baseline body, --min-syntax-rate 0.92 (forces
# iteration), --min-accuracy 0.31, --max-iterations 15, Sonnet-4-6 author thinking-high,
# seed429 TRAIN.
set -uo pipefail
cd /home/aadivyar/csd-generation

export CUDA_VISIBLE_DEVICES=1,2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

OUT=outputs/generated/ralph_1p5B_gsm_h5_postcondition_20260609
GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed429.json
LOG=/tmp/ralph_1p5B_gsm_h5_postcondition_20260609.log

mkdir -p "$OUT"

nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_1p5B_gsm_h5_postcondition_20260609 \
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
