#!/bin/bash
# Phase 2: re-launch Spider-14B-Instruct reeval at N=100 on GPU 2
# AFTER the 4-prior-wins chain finishes (the chain holds GPU 2 until then).
# Same args as the original launch_spider14b_synth_reeval_N100_20260601.sh,
# but waits for the chain process group first.
set -u
cd /home/aadivyar/csd-generation
mkdir -p logs/reeval outputs/reeval
LOG=/home/aadivyar/csd-generation/logs/reeval/spider14b_synth_reeval_N100_20260601.log
CHAIN_LOG=/home/aadivyar/csd-generation/logs/reeval/prior_wins_reeval_chain_20260601.log
echo "[phase2 $(date)] waiting for chain to finish" >> "$CHAIN_LOG"
while pgrep -f "launch_prior_wins_reeval_N100_chain_20260601.sh" >/dev/null 2>&1 \
   || pgrep -f "reevaluate_compiled_csd .*ralph_(1p5B|7B)_(gsm|spider)_" >/dev/null 2>&1; do
  sleep 30
done
sleep 20
echo "[phase2 $(date)] GPU 2 free, launching Spider-14B reeval" >> "$CHAIN_LOG"
COMPILED=/home/aadivyar/csd-generation/outputs/generated/ralph_14B_spider_instruct_win_20260601/ralph_14B_spider_instruct_win_20260601_20260601_113141_82910d/python/ralph_14B_spider_instruct_win_20260601/GeneratedCSD.py
OUT=/home/aadivyar/csd-generation/outputs/reeval/ralph_14B_spider_instruct_win_20260601_N100.json
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES=2 /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.scripts.reevaluate_compiled_csd \
  "$COMPILED" \
  --dataset spider \
  --eval-model Qwen/Qwen2.5-14B-Instruct --eval-backend vllm \
  --sample-size 100 --max-steps 600 --step-token-budget 1 \
  --vllm-gpu-memory-utilization 0.85 --vllm-tensor-parallel-size 1 \
  --max-seconds-per-example 180 \
  --output-json "$OUT" \
  > "$LOG" 2>&1
echo "[phase2 $(date)] spider14b reeval done exit=$?" >> "$CHAIN_LOG"
