#!/bin/bash
# GSM-1.5B-Instruct disjoint-split RETRY at max-iterations=30, 2026-06-03.
# Prior iter-20 run (ralph_1p5B_gsm_disjoint_20260602) saved a fallback at
# attempt 12 acc=0.408 / syn=0.449 — accuracy clears 0.32 bar but syntax
# misses the 0.90 bar. User approved retry with more iterations.
# Synthesis config IDENTICAL to launch_gsm1p5b_disjoint_20260602.sh except:
#   - --max-iterations 20 -> 30
#   - output dir/name/log relabeled with _iter30_20260603 suffix
# Thresholds (0.32 / 0.90), task string, prompt, library, split all unchanged.
# Sonnet-4.6 author. Runs on GPU 2.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_gsm_disjoint_iter30_20260603
mkdir -p "$OUT"
GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 30 \
  --output-name ralph_1p5B_gsm_disjoint_iter30_20260603 \
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
  > /tmp/ralph_1p5B_gsm_disjoint_iter30_20260603.log 2>&1 &
echo "GSM-1.5B-DISJOINT-iter30 pid: $!  -> $OUT  log=/tmp/ralph_1p5B_gsm_disjoint_iter30_20260603.log"
