#!/bin/bash
# GSM-14B-Instruct metaDecode WIN run 2026-06-01.
# Mirrors launch_gsm7b_instruct_win_20260601.sh EXACTLY except:
#   - eval-model:                  Qwen2.5-7B-Instruct  -> Qwen2.5-14B-Instruct
#   - vllm-gpu-memory-utilization: 0.45                 -> 0.85 (14B fp16 needs ~28GB on 40GB card)
#   - output dir/name + log path
#   - --min-accuracy / --min-syntax-rate placeholders to be patched in once
#     Phase A baseline numbers land (--min-accuracy = CRANE-14B acc + 0.01,
#     --min-syntax-rate kept at 0.90 per memory feedback_no_min_syntax_100).
# Sonnet-4.6 authors the strategy (thinking=high). NO strategy guidance in prompts.
# Initial placeholder thresholds (0.45 / 0.90) reuse the 7B floor; the supervisor
# will rewrite them before launch if A.1 lands a higher CRANE-14B acc.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_14B_gsm_instruct_win_20260601
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=__GSM_GPU__ LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-14B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_14B_gsm_instruct_win_20260601 \
  --min-accuracy __GSM_MIN_ACC__ --min-syntax-rate 0.90 \
  --eval-sample-size 50 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 120 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.85 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json \
  --gsm-split-name eval \
  > /tmp/ralph_14B_gsm_instruct_win_20260601.log 2>&1 &
echo "GSM-14B-Instruct WIN pid: $!  -> $OUT  log=/tmp/ralph_14B_gsm_instruct_win_20260601.log"
