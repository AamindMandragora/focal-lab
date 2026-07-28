#!/bin/bash
set -e
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
OUT=/home/aadivyar/csd-generation/outputs/abtest_appendtaskguidance
for ARM in A_original B_empty C_antifmt; do
  echo "=== START $ARM $(date) ==="
  python -m synthesis.scripts.reevaluate_compiled_csd \
    $OUT/$ARM/GeneratedCSD.py \
    --dataset gsm_symbolic \
    --eval-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --eval-backend vllm \
    --device cuda \
    --sample-size 25 \
    --max-steps 900 \
    --step-token-budget 1 \
    --vllm-gpu-memory-utilization 0.5 \
    --vllm-tensor-parallel-size 1 \
    --output-json $OUT/$ARM/result.json \
    2>&1 | tee $OUT/$ARM/eval.log
  echo "=== END $ARM $(date) ==="
done
echo ALL DONE
