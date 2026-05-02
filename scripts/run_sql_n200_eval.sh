#!/bin/bash
set -e
cd ~/csd-generation
source ~/.profile
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CUDA_VISIBLE_DEVICES=2 /opt/anaconda/bin/python -m evaluations.sql_spider.cli \
    --run-dir /home/aadivyar/csd-generation/outputs/generated-csd/runs/20260428_044720_113b97 \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --backend vllm \
    --device cuda \
    --limit 200 \
    --random-sample \
    --seed 123 \
    --max-steps 200 \
    --vllm-tensor-parallel-size 1 \
    --vllm-max-model-len 4096 \
    --vllm-gpu-memory-utilization 0.40
