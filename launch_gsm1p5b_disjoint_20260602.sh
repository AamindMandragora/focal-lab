#!/bin/bash
# GSM-1.5B-Instruct disjoint-split win retry, 2026-06-02.
# Prior N=50 win (ralph_1p5B_gsm_relaunch_20260530_fixedlib, 42/96) failed N=100
# reeval (45/44 — accuracy held, syntax collapsed). Same family of root cause:
# train==eval at N=50 with strategy effective only on those exact 50 examples.
# This run uses the 49-example TRAIN half of gsm_symbolic_crane_proportional_49x49_seed123.json
# (disjoint from the 49-example HELD-OUT half). Population only allows 49/49
# disjoint, not 50/50 — one bucket would round up beyond population.
# Synthesis config unchanged vs launch_gsm1p5b_relaunch_20260530_fixedlib.sh except:
#   - --gsm-split-file switched to the new 49x49 file; split-name stays "train"
#   - --eval-sample-size 50 -> 49; min-examples-before-threshold-stop 50 -> 49
#   - output dir/name/log relabeled
# Thresholds (0.32 / 0.90), task string, prompt, library all unchanged. Sonnet-4.6 author.
# GPU 2 (40GB free). util 0.40 ~= 16GB.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_gsm_disjoint_20260602
mkdir -p "$OUT"
GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name ralph_1p5B_gsm_disjoint_20260602 \
  --min-accuracy 0.32 --min-syntax-rate 0.90 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.40 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file "$GSM_SPLIT" --gsm-split-name train \
  > /tmp/ralph_1p5B_gsm_disjoint_20260602.log 2>&1 &
echo "GSM-1.5B-DISJOINT pid: $!  -> $OUT  log=/tmp/ralph_1p5B_gsm_disjoint_20260602.log"
