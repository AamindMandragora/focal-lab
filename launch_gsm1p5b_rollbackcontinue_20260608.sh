#!/bin/bash
# H2 experiment: GSM-1.5B syntax-rate run with RollbackAndContinue available to the author.
# Mirrors the proven seed429 warm-start run; the ONLY deliberate changes are:
#   - start from the clean 28/43 baseline body (gsm1p5b_seed429_warmstart_body.dfy)
#   - RollbackAndContinue tool now deployed (the single variable under test)
#   - --max-iterations 15, GPUs 1,2, new output name
# Optimizes on seed429 TRAIN (disjoint from the held-out seed123 EVAL).
set -uo pipefail
cd /home/aadivyar/csd-generation

export CUDA_VISIBLE_DEVICES=1,2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

OUT=outputs/generated/ralph_1p5B_gsm_rollbackcontinue_20260608
GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed429.json
LOG=/tmp/ralph_1p5B_gsm_rollbackcontinue_20260608.log

mkdir -p "$OUT"

nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_1p5B_gsm_rollbackcontinue_20260608 \
  --output-dir "$OUT" \
  --min-accuracy 0.31 --min-syntax-rate 0.85 \
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
