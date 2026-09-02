#!/bin/bash
# Spider-1.5B relaunch (2026-05-29). Mirrors launch_spider1p5b_delimhint.sh:
# flat prompt delivery (multi-turn zeroes the Spider constrained eval), the
# sharpened "force the delimiter" feedback hint (live in feedback_loop.py), and
# the 0.52/0.85 win bars. ONLY operational change vs delimhint: vLLM memory
# fraction 0.40 -> 0.20 because GPU 1 currently has ~10GB free (shared card).
# That just shrinks the KV cache; it changes no experimental parameter.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_20260529_relaunch_spider
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 10 \
  --output-name metadecode_spider_Qwen_Qwen2.5_1.5B_Instruct_sonnet4.6_iter10_tb1_ms600 \
  --min-accuracy 0.52 --min-syntax-rate 0.85 \
  --eval-sample-size 50 --eval-max-steps 600 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  > /tmp/ralph_1p5B_20260529_relaunch_spider.log 2>&1 &
echo "Spider-1.5B pid: $!  -> $OUT"
