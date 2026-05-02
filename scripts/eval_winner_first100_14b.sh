#!/bin/bash
set -e
cd ~/csd-generation
source ~/.profile
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Re-evaluate the n=100 winning strategy on the FIRST 100 Spider problems
# (matching IterGen sample selection) with Qwen2.5-Coder-14B-Instruct.

CUDA_VISIBLE_DEVICES=0 SQL_PRED_DUMP=/tmp/our_first100_14b_dump.json /opt/anaconda/bin/python -m evaluations.sql_spider.cli \
    --run-dir /home/aadivyar/csd-generation/outputs/generated-csd/runs/20260429_092049_eb1a4c \
    --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --backend vllm \
    --device cuda \
    --limit 100 \
    --max-steps 400 \
    --vllm-tensor-parallel-size 1 \
    --vllm-max-model-len 4096 \
    --vllm-gpu-memory-utilization 0.75
