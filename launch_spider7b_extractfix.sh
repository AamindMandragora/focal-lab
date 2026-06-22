#!/bin/bash
# Spider-7B relaunch AFTER the << >> span-extraction fix (regex [^<>] -> .*? in
# evaluator._extract_constrained_content + delimited_output.DELIMITED_SPAN_PATTERN).
# Before the fix, valid SQL containing comparison operators (age < 30, COUNT(*) > 1)
# was extracted as ZERO spans -> auto syntax-fail, capping syntax at ~80%.
# Offline re-score of the prior best attempt: 72% acc / 98% syntax with the fix,
# which clears both bars (>=0.71 acc, >=0.90 syntax). This run gets it officially accepted.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_7B_20260527_extractfix_spider
mkdir -p "$OUT"
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
  > /tmp/ralph_7B_20260527_extractfix_spider.log 2>&1 &
echo "Spider-7B pid: $!  -> $OUT"
