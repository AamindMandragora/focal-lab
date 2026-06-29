#!/bin/bash
# GSM-7B synthesis with the new trade-offs prompt, on GPU2.
# Matches ralph_7B_20260527_065955's GSM config exactly except output-dir.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_7B_20260527_libstdcfix_gsmnewprompt
mkdir -p "$OUT"
# Prepend the conda env's own lib dir so the loader prefers its newer libstdc++
# (has CXXABI_1.3.15, which scipy's HiGHS solver needs). Calling the env's python
# binary directly skips `conda activate`, which is what normally sets this.
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-Coder-7B-Instruct --eval-backend vllm \
  --max-iterations 10 \
  --output-name metadecode_gsm_symbolic_Qwen_Qwen2.5_Coder_7B_Instruct_sonnet4.6_iter10_tb1_ms900 \
  --min-accuracy 0.25 --min-syntax-rate 0.9 \
  --eval-sample-size 50 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.55 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json \
  --gsm-split-name eval \
  > /tmp/ralph_7B_20260527_libstdcfix_gsmnewprompt.log 2>&1 &
echo "GSM-7B pid: $!  -> $OUT"
