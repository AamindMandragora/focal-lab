#!/bin/bash
# Spider-1.5B max-steps bump (2026-05-29). Identical to launch_spider1p5b_relaunch.sh
# EXCEPT the step/time budget: the relaunch run hit the delimiter CLOSE wall --
# 15-36/50 outputs opened `<<` but the *step budget* (600) ran out before the
# strategy could reach a complete SQL prefix and emit the closing `>>`. This run
# doubles the per-example step budget (600 -> 1200) so a forced/natural close has
# room to fire, and raises the per-example wall-clock cap (90s -> 180s) so the
# time cap cannot silently preempt the larger step budget (otherwise the bump
# would test nothing). No experimental parameter, prompt, or code changes.
# OPERATIONAL ONLY: runs on GPU 3 (most free, ~8GB) at vLLM util 0.17 because
# GPU 1 is currently pinned by other tenants; util just sizes the KV cache.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_20260529_spider_maxsteps1200
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 10 \
  --output-name metadecode_spider_Qwen_Qwen2.5_1.5B_Instruct_sonnet4.6_iter10_tb1_ms1200 \
  --min-accuracy 0.52 --min-syntax-rate 0.85 \
  --eval-sample-size 50 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.17 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  > /tmp/ralph_1p5B_20260529_spider_maxsteps1200.log 2>&1 &
echo "Spider-1.5B ms1200 pid: $!  -> $OUT"
