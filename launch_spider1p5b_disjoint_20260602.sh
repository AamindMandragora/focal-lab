#!/bin/bash
# Spider-1.5B-Instruct disjoint-split win retry, 2026-06-02.
# The prior N=50 win (ralph_1p5B_spider_win_20260531) failed N=100 reeval
# (24/81), suggesting overfit to the first 50 examples. This run tunes on
# 100 NEW Spider examples (the train half of spider_dev_proportional_100x100_seed123.json)
# and the held-out 100 (the test half) is reserved for the final apples-to-apples
# reeval.
# Synthesis config is unchanged vs launch_spider1p5b_win_20260531.sh except:
#   - --spider-split-file/--spider-split-name train added (NEW train half, disjoint from held-out)
#   - --eval-sample-size 50 -> 100 (use the full train half)
#   - --eval-min-examples-before-threshold-stop 50 -> 100
#   - output dir/name/log relabeled "disjoint_20260602"
# Thresholds (0.42 / 0.88) and prompt/code unchanged. Sonnet-4.6 author.
# GPU choice: 2 (40GB free). util 0.20 (~8GB) — small 1.5B, room to spare.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_spider_disjoint_20260602
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_100x100_seed123.json
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name ralph_1p5B_spider_disjoint_20260602 \
  --min-accuracy 0.42 --min-syntax-rate 0.88 \
  --eval-sample-size 100 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 100 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  > /tmp/ralph_1p5B_spider_disjoint_20260602.log 2>&1 &
echo "Spider-1.5B-DISJOINT pid: $!  -> $OUT  log=/tmp/ralph_1p5B_spider_disjoint_20260602.log"
