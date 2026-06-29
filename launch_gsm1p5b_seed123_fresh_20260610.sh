#!/bin/bash
# Fresh GSM-1.5B metaDecode synthesis on seed123 TRAIN (no seed429 warmstart — contamination rule).
# Bar from controlled baselines (seed123 eval, non-Coder): crane 0.388 is max -> --min-accuracy 0.41.
set -uo pipefail
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}

python -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters." \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name gsm1p5b_seed123_fresh_20260610 \
  --output-dir outputs/generated/gsm1p5b_seed123_fresh_20260610 \
  --min-accuracy 0.41 --min-syntax-rate 0.90 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 120 \
  --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.30 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --refinement-beam-size 2 \
  --gsm-split-file environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json \
  --gsm-split-name train
echo "EXIT_SYNTH_GSM1P5B=$?"
echo DONE_GSM1P5B_SYNTH
