#!/bin/bash
# GSM-1.5B relaunch on GPU2 under the SHARPENED delimiter feedback (2026-05-28).
# Identical to launch_gsm1p5b_graderfix.sh except OUT/log tags -> only changed
# variable is the new "force the delimiter" diagnostic in feedback_loop.py.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_1p5B_20260528_delimhint_gsm
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 10 \
  --output-name metadecode_gsm_symbolic_Qwen_Qwen2.5_1.5B_Instruct_sonnet4.6_iter10_tb1_ms900 \
  --min-accuracy 0.32 --min-syntax-rate 0.44 \
  --eval-sample-size 50 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.40 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json \
  --gsm-split-name eval \
  > /tmp/ralph_1p5B_20260528_delimhint_gsm.log 2>&1 &
echo "GSM-1.5B pid: $!  -> $OUT"
