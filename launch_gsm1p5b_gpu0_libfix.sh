#!/bin/bash
# GSM-1.5B relaunch on GPU0 (only free GPU; 1/2/3 occupied by others).
# Fixes the libstdc++ CXXABI_1.3.15 eval crash that wiped out the previous
# hinfix run (every attempt got 0/0 examples) by prepending the conda lib dir
# to LD_LIBRARY_PATH. Fresh process also picks up the fixed feedback_loop.py.
# Dedicated GPU (mem util 0.40) so eval compute isn't shared -> reliable timing.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_gsm_libfix_v3
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_1p5B_gsm_libfix_v3 \
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
  > /tmp/ralph_1p5B_gsm_libfix_v3.log 2>&1 &
echo "GSM-1.5B pid: $!  -> $OUT  log=/tmp/ralph_1p5B_gsm_libfix_v3.log"
