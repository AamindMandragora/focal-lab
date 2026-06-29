#!/bin/bash
# GSM-1.5B synthesis with the new trade-offs prompt, on GPU1.
# Waits for the in-flight Spider-1.5B run (wrapper PID 4109984) to free GPU1, then launches.
# Matches ralph_1p5B_20260527_065955's GSM config exactly except output-dir.
set -u
WAIT_PID=4109984
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[queue] GPU1 freed (PID $WAIT_PID gone) at $(date)"
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_20260527_libstdcfix_gsmnewprompt
mkdir -p "$OUT"
# Prepend the conda env's own lib dir so the loader prefers its newer libstdc++
# (has CXXABI_1.3.15, which scipy's HiGHS solver needs). Calling the env's python
# binary directly skips `conda activate`, which is what normally sets this.
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 10 \
  --output-name metadecode_gsm_symbolic_Qwen_Qwen2.5_1.5B_Instruct_sonnet4.6_iter10_tb1_ms900 \
  --min-accuracy 0.39 --min-syntax-rate 0.44 \
  --eval-sample-size 50 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.30 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json \
  --gsm-split-name eval \
  > /tmp/ralph_1p5B_20260527_libstdcfix_gsmnewprompt.log 2>&1
echo "[queue] GSM-1.5B finished at $(date) -> $OUT"
