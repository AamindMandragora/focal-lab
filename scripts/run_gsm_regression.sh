#!/bin/bash
set -e
cd ~/csd-generation
source ~/.profile
export VLLM_WORKER_MULTIPROC_METHOD=spawn

TASK_DESC="Solve GSM-Symbolic math word problems. Natural-language reasoning may stay outside constrained spans; for scoring, the required constrained region is the final answer only. The last <<...>> span must contain one valid symbolic arithmetic expression that answers the question. Do not require every intermediate calculation to be wrapped."

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
