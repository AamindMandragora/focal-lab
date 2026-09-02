#!/bin/bash
# Spider-14B-Instruct metaDecode WIN run 2026-06-01.
# Mirrors launch_spider7b_win_20260531.sh EXACTLY except:
#   - eval-model:                  Qwen2.5-7B-Instruct  -> Qwen2.5-14B-Instruct
#   - vllm-gpu-memory-utilization: 0.50                 -> 0.85
#   - output dir/name + log path
#   - --min-accuracy / --min-syntax-rate placeholders patched after Phase A.3/A.4.
# Sonnet-4.6 authors the strategy (thinking=high). NO strategy guidance in prompts.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_14B_spider_instruct_win_20260601
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-14B-Instruct --eval-backend vllm \
  --max-iterations 10 \
  --output-name ralph_14B_spider_instruct_win_20260601 \
  --min-accuracy 0.47 --min-syntax-rate 0.90 \
  --eval-sample-size 50 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.85 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  > /home/aadivyar/csd-generation/logs/baselines_14B/ralph_14B_spider_instruct_win_20260601.log 2>&1 &
echo "Spider-14B-Instruct WIN pid: $!  -> $OUT  log=/home/aadivyar/csd-generation/logs/baselines_14B/ralph_14B_spider_instruct_win_20260601.log"
