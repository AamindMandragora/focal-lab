#!/bin/bash
# GSM-7B-Instruct disjoint-split win retry, 2026-06-02.
# Prior N=50 win (ralph_7B_gsm_instruct_win_20260601_v2, 60/96) failed N=100
# reeval (62/92, contains_delimiters=False — syntax 5-8pp below baseline).
# Same overfit-to-first-50 family.
# This run uses the 49-example TRAIN half of gsm_symbolic_crane_proportional_49x49_seed123.json.
# Synthesis config unchanged vs launch_gsm7b_instruct_win_20260601.sh except:
#   - --gsm-split-file switched to the new 49x49 file; split-name "train"
#   - --eval-sample-size 50 -> 49; min-examples-before-threshold-stop 50 -> 49
#   - output dir/name/log relabeled
# Thresholds (0.45 / 0.96), task string, prompt, library all unchanged. Sonnet-4.6 author.
# GPU 2 sequentially after Spider-7B (40GB free). util 0.45 ~= 18GB.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_7B_gsm_disjoint_20260602
mkdir -p "$OUT"
GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name ralph_7B_gsm_disjoint_20260602 \
  --min-accuracy 0.45 --min-syntax-rate 0.96 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.45 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file "$GSM_SPLIT" --gsm-split-name train \
  > /tmp/ralph_7B_gsm_disjoint_20260602.log 2>&1 &
echo "GSM-7B-DISJOINT pid: $!  -> $OUT  log=/tmp/ralph_7B_gsm_disjoint_20260602.log"
