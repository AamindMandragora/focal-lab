#!/bin/bash
set -e
cd ~/csd-generation
source ~/.profile
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Run 2 / attempt 6 (63% on seed=123, uses RollbackToBoundary with ',' + '(' boundaries).
# Re-evaluate on the SAME first-100 / 14B-Instruct that IterGen scored 73.0% on,
# with --max-steps 400 to address the truncation bottleneck.

CUDA_VISIBLE_DEVICES=0 SQL_PRED_DUMP=/tmp/attempt6_first100_14b_dump.json /opt/anaconda/bin/python -m evaluations.sql_spider.cli \
    --run-dir /home/aadivyar/csd-generation/outputs/generated-csd/runs/20260429_205530_df0cae/sql_validgroups_n100_csd_20260429_213930_68bd60 \
    --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --backend vllm \
    --device cuda \
    --limit 100 \
    --max-steps 400 \
    --vllm-tensor-parallel-size 1 \
    --vllm-max-model-len 4096 \
    --vllm-gpu-memory-utilization 0.75
