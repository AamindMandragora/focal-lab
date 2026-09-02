#!/bin/bash
# GSM-7B-Instruct metaDecode win run 2026-06-01.
# Mirrors the GSM-1.5B winning config (ralph_1p5B_gsm_relaunch_20260530_fixedlib)
# but retargeted to the non-Coder Qwen2.5-7B-Instruct and the in-house 7B bar:
#   CRANE 44% acc / 97% syntax  AND  IterGen 41% acc / 100% syntax (N=100, 06-01).
#   Combined win = acc > 44% AND syntax >= 100%.
# min-accuracy 0.45 is the just-above-CRANE threshold. min-syntax-rate 0.90 is
# intentionally below 1.0 — per project memory, launching with 1.0 cuts evals at
# 1-3 examples on the threshold-impossible early-stop. The final win check is
# done by re-scoring on the full 50-sample failure_report.
# GPU 2 (40 GB free at launch); 0.45 utilization ~= 18 GB, leaves headroom.
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_7B_gsm_instruct_win_20260601_v2
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_7B_gsm_instruct_win_20260601_v2 \
  --min-accuracy 0.45 --min-syntax-rate 0.96 \
  --eval-sample-size 50 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.45 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json \
  --gsm-split-name eval \
  > /tmp/ralph_7B_gsm_instruct_win_20260601_v2.log 2>&1 &
echo "GSM-7B Instruct v2 pid: $!  -> $OUT  log=/tmp/ralph_7B_gsm_instruct_win_20260601_v2.log"
