#!/bin/bash
set -u

cd /home/aadivyar/csd-generation

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to a free GPU index}"

GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed429.json
OUT=outputs/generated/ralph_1p5B_gsm_fresh_iter30_20260608_rerun1
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} \
nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems. Keep all intermediate reasoning outside visible spans. Use exactly one visible << >> span for the final answer only, never put prose inside that visible span, and open it only when the final answer is known.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 30 \
  --output-name ralph_1p5B_gsm_fresh_iter30_20260608_rerun1 \
  --min-accuracy 0.32 --min-syntax-rate 0.95 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 120 --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 --restart-after-stuck-iters 1 \
  --vllm-gpu-memory-utilization 0.18 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --helper-mask-min-evals 2 --helper-mask-min-uses 2 --helper-mask-margin 0.2 --helper-mask-max-disabled 8 \
  --helper-bandit-min-evals 2 --helper-bandit-top-k 12 --helper-bandit-ucb-c 0.3 --helper-bandit-explore-untried 1 \
  --refinement-beam-size 4 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file "$GSM_SPLIT" --gsm-split-name train \
  > /tmp/ralph_1p5B_gsm_fresh_iter30_20260608_rerun1.log 2>&1 &

echo "GSM-1.5B fresh iter30 pid: $!  -> $OUT  log=/tmp/ralph_1p5B_gsm_fresh_iter30_20260608_rerun1.log"
