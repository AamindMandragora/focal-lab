#!/bin/bash
# GSM-1.5B relaunch 2026-05-29 (corrected win bar) on freed GPU 1.
# Identical to launch_gsm1p5b_relaunch_20260529.sh EXCEPT the syntax bar and the
# output dir/name/log. The prior relaunch declared success at attempt 8 with only
# 52% syntax because --min-syntax-rate 0.44 was too lenient; the user clarified a
# win must beat the PUBLISHED CRANE result (31% acc / ~100% syntax) on BOTH axes.
# So the syntax bar is raised 0.44 -> 0.90 ("strong", user-chosen) to force the
# synthesis loop to keep iterating toward near-published syntax. Accuracy bar stays
# 0.32 (beats published CRANE 31%). New output dir keeps the rejected run intact.
# OPERATIONAL ONLY: no experimental parameter, prompt, or code changes.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_gsm_relaunch_20260529_syn090
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_1p5B_gsm_relaunch_20260529_syn090 \
  --min-accuracy 0.32 --min-syntax-rate 0.90 \
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
  > /tmp/ralph_1p5B_gsm_relaunch_20260529_syn090.log 2>&1 &
echo "GSM-1.5B syn090 pid: $!  -> $OUT  log=/tmp/ralph_1p5B_gsm_relaunch_20260529_syn090.log"
