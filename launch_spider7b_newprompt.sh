#!/bin/bash
# Spider-7B synthesis with the new trade-offs prompt, on GPU2 (freed by the GSM-7B win).
# Config copied from the most recent sonnet4.6 Spider-7B run
# (metadecode_spider_..._iter3_..._20260526_163809) but with 10 iterations and the
# libstdc++ eval fix + full-sample threshold-stop guard, matching launch_gsm7b_newprompt.sh.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_7B_20260527_libstdcfix_spidernewprompt
mkdir -p "$OUT"
# Prepend the conda env's own lib dir so the loader prefers its newer libstdc++
# (has CXXABI_1.3.15, which scipy's HiGHS solver needs). Calling the env's python
# binary directly skips `conda activate`, which is what normally sets this.
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-Coder-7B-Instruct --eval-backend vllm \
  --max-iterations 10 \
  --output-name metadecode_spider_Qwen_Qwen2.5_Coder_7B_Instruct_sonnet4.6_iter10_tb1_ms600 \
  --min-accuracy 0.71 --min-syntax-rate 0.9 \
  --eval-sample-size 50 --eval-max-steps 600 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.55 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  > /tmp/ralph_7B_20260527_libstdcfix_spidernewprompt.log 2>&1 &
echo "Spider-7B pid: $!  -> $OUT"
