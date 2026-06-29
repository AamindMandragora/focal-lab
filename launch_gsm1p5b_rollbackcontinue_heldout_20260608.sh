#!/bin/bash
# Held-out validation of the strategy produced by the H2 RollbackAndContinue run.
# This is the MEASUREMENT step of the already-approved H2 run, not a new experiment:
#   - evaluate the authored body ONCE (--max-iterations 1) on the protected held-out split
#   - seed123 EVAL split (N=49), disjoint from the seed429 TRAIN it was optimized on
#   - eval model Qwen2.5-1.5B-Instruct, same eval flags as the H2 run
# Low min thresholds so the runner accepts the provided strategy and reports its eval
# (no spurious author/refinement call at max-iterations 1).
set -uo pipefail
cd /home/aadivyar/csd-generation

export CUDA_VISIBLE_DEVICES=1,2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

OUT=outputs/generated/ralph_1p5B_gsm_rollbackcontinue_heldout_20260608
GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
BODY=/home/aadivyar/csd-generation/gsm1p5b_rollbackcontinue_authored_body.dfy
LOG=/tmp/ralph_1p5B_gsm_rollbackcontinue_heldout_20260608.log

mkdir -p "$OUT"

nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 1 \
  --output-name ralph_1p5B_gsm_rollbackcontinue_heldout_20260608 \
  --output-dir "$OUT" \
  --min-accuracy 0.0 --min-syntax-rate 0.0 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 120 --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.18 --device auto \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file "$GSM_SPLIT" --gsm-split-name eval \
  --initial-strategy-file "$BODY" \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
echo "OUT=$OUT"
