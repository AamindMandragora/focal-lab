#!/bin/bash
# Re-evaluate the Spider-14B-Instruct metaDecode WIN strategy at N=100
# (apples-to-apples vs IterGen 14B 0.46/1.00 baseline at N=100).
# Strategy file is the GeneratedCSD.py from the successful Phase B.2 synth
# (ralph_14B_spider_instruct_win_20260601, attempt 1, accuracy 0.66 / syntax 0.98 @ N=50).
set -u
cd /home/aadivyar/csd-generation
COMPILED=/home/aadivyar/csd-generation/outputs/generated/ralph_14B_spider_instruct_win_20260601/ralph_14B_spider_instruct_win_20260601_20260601_113141_82910d/python/ralph_14B_spider_instruct_win_20260601/GeneratedCSD.py
OUT=/home/aadivyar/csd-generation/outputs/reeval/ralph_14B_spider_instruct_win_20260601_N100.json
mkdir -p "$(dirname "$OUT")"
mkdir -p /home/aadivyar/csd-generation/logs/reeval
LOG=/home/aadivyar/csd-generation/logs/reeval/spider14b_synth_reeval_N100_20260601.log
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES=2 nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.scripts.reevaluate_compiled_csd \
  "$COMPILED" \
  --dataset spider \
  --eval-model Qwen/Qwen2.5-14B-Instruct --eval-backend vllm \
  --sample-size 100 --max-steps 600 --step-token-budget 1 \
  --vllm-gpu-memory-utilization 0.85 --vllm-tensor-parallel-size 1 \
  --max-seconds-per-example 180 \
  --output-json "$OUT" \
  > "$LOG" 2>&1 &
echo "spider14b reeval pid: $!  out=$OUT  log=$LOG"
