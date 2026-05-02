#!/bin/bash
set -e
cd ~/csd-generation
source ~/.profile
export VLLM_WORKER_MULTIPROC_METHOD=spawn

TASK_DESC="Solve math word problems step by step, writing each arithmetic computation inside << >> delimiters."

CUDA_VISIBLE_DEVICES=0 /opt/anaconda/bin/python run_synthesis.py \
    --task "$TASK_DESC" \
    --dataset gsm_symbolic \
    --max-iterations 15 \
    --generation-model "gpt-5.4" \
    --generation-backend openai \
    --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --eval-backend vllm \
    --output-name "gsm_regression_csd" \
    --temperature 0.7 \
    --device cuda \
    --min-accuracy 0.30 \
    --min-syntax-rate 0.50 \
    --eval-sample-size 10 \
    --eval-seed 123 \
    --eval-max-steps 256 \
    --vllm-tensor-parallel-size 1 \
    --vllm-max-model-len 4096 \
    --vllm-gpu-memory-utilization 0.40 \
    --synthesis-max-tokens 2048
