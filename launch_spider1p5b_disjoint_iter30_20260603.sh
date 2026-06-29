#!/bin/bash
# Spider-1.5B-Instruct disjoint-split RETRY at max-iterations=30, 2026-06-03.
# Prior iter-20 run (ralph_1p5B_spider_disjoint_20260602) saved no fallback —
# no attempt cleared the 0.42 accuracy bar. User approved retry with more
# iterations.
# Synthesis config IDENTICAL to launch_spider1p5b_disjoint_20260602.sh except:
#   - --max-iterations 20 -> 30
#   - output dir/name/log relabeled with _iter30_20260603 suffix
# Thresholds (0.42 / 0.88), task string, prompt, library, split all unchanged.
# Sonnet-4.6 author. Runs on GPU 2.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_spider_disjoint_iter30_20260603
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_100x100_seed123.json
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 30 \
  --output-name ralph_1p5B_spider_disjoint_iter30_20260603 \
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
  > /tmp/ralph_1p5B_spider_disjoint_iter30_20260603.log 2>&1 &
echo "Spider-1.5B-DISJOINT-iter30 pid: $!  -> $OUT  log=/tmp/ralph_1p5B_spider_disjoint_iter30_20260603.log"
