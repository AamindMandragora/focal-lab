#!/bin/bash
# Spider-1.5B (CODER) re-confirm 2026-05-30 — FIXED-LIBRARY run.
# Purpose: re-confirm the Spider-1.5B win on the apples-to-apples CRANE model
# (Qwen2.5-Coder-1.5B), under the CURRENT library + patched grader, at the
# corrected syntax bar. Best prior result on this model is 50%/92% (05-18), which
# was "rejected" ONLY under the old 94% syntax threshold; under the 0.90 bar it is
# a win vs published CRANE-Coder-1.5B = 32% acc / 100% syntax.
# Mirrors launch_spider1p5b_relaunch_20260529.sh EXCEPT:
#   1. eval-model Qwen/Qwen2.5-1.5B-Instruct -> Qwen/Qwen2.5-Coder-1.5B-Instruct
#      (the model the only CRANE Spider-1.5B baseline is measured on).
#   2. min-accuracy 0.52 -> 0.34 (strictly beat CRANE 32%: 17/50 = 0.34 > 0.32),
#      min-syntax 0.85 -> 0.90 (so "success" == a real win).
#   3. GPU 2 (util 0.17 ~6.8GB; 23.3GB free there, safe neighbor).
# Runs on the fixed CloseConstrainedSpan library already in the synthesis tree.
# Keeps the proven Spider-1.5B template: util 0.17, ms1200, 180s wall, multi-turn
# prompt (current code), backtick task string verbatim. New output dir/name/log.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_spider_coder_relaunch_20260530_fixedlib
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-Coder-1.5B-Instruct --eval-backend vllm \
  --max-iterations 10 \
  --output-name ralph_1p5B_spider_coder_relaunch_20260530_fixedlib \
  --min-accuracy 0.34 --min-syntax-rate 0.90 \
  --eval-sample-size 50 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 180 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.17 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  > /tmp/ralph_1p5B_spider_coder_relaunch_20260530_fixedlib.log 2>&1 &
echo "Spider-1.5B-Coder fixedlib pid: $!  -> $OUT  log=/tmp/ralph_1p5B_spider_coder_relaunch_20260530_fixedlib.log"
