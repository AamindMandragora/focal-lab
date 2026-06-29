#!/bin/bash
# GSM-1.5B relaunch 2026-05-29 on freed GPU 1 (orphans cleared; GPU 0=nalin,
# 2=baseline/free, 3=gemini Spider-7B ablation). Mirrors launch_gsm1p5b_gpu0_libfix.sh
# EXACTLY except GPU index, output dir, and log path. Keeps the libstdc++
# CXXABI_1.3.15 LD_LIBRARY_PATH fix and the grader-fixed feedback_loop.py.
# OPERATIONAL ONLY: no experimental parameter, prompt, or code changes.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_gsm_relaunch_20260529
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_1p5B_gsm_relaunch_20260529 \
  --min-accuracy 0.32 --min-syntax-rate 0.44 \
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
  > /tmp/ralph_1p5B_gsm_relaunch_20260529.log 2>&1 &
echo "GSM-1.5B pid: $!  -> $OUT  log=/tmp/ralph_1p5B_gsm_relaunch_20260529.log"
