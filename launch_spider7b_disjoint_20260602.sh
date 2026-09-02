#!/bin/bash
# Spider-7B-Instruct disjoint-split win retry, 2026-06-02.
# Prior N=50 win (ralph_7B_spider_win_20260531) failed N=100 reeval (33/73 with
# contains_delimiters=False). Same root cause as 1.5B: overfit to first 50.
# This run tunes on the 100-example train half (disjoint from the 100-example
# held-out half) of spider_dev_proportional_100x100_seed123.json.
# Synthesis config unchanged vs launch_spider7b_win_20260531.sh except:
#   - --spider-split-file/--spider-split-name train added
#   - --eval-sample-size 50 -> 100; min-examples-before-threshold-stop 50 -> 100
#   - output dir/name/log relabeled
# Thresholds (0.64 / 0.93) and prompt/code unchanged. Sonnet-4.6 author.
# GPU 2 sequentially after 1.5B (40GB free at launch). util 0.50 ~= 20GB.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_7B_spider_disjoint_20260602
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_100x100_seed123.json
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name ralph_7B_spider_disjoint_20260602 \
  --min-accuracy 0.64 --min-syntax-rate 0.93 \
  --eval-sample-size 100 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 100 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.50 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  > /tmp/ralph_7B_spider_disjoint_20260602.log 2>&1 &
echo "Spider-7B-DISJOINT pid: $!  -> $OUT  log=/tmp/ralph_7B_spider_disjoint_20260602.log"
