#!/bin/bash
set -e
cd ~/csd-generation
source ~/.profile
export VLLM_WORKER_MULTIPROC_METHOD=spawn

TASK_DESC="Text-to-SQL generation on the Spider benchmark. The model reads a schema (tables and columns) and a natural-language question, then emits a single SQL query as its output. The parser validates the query against a SQL grammar that is dynamically narrowed to the current schema's tables and columns."

CUDA_VISIBLE_DEVICES=1 /opt/anaconda/bin/python run_synthesis.py \
    --task "$TASK_DESC" \
    --dataset spider \
    --max-iterations 20 \
    --generation-model "gpt-5.4" \
    --generation-backend openai \
    --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --eval-backend vllm \
    --output-name "sql_validgroups_n100_csd" \
    --temperature 0.7 \
    --device cuda \
    --min-accuracy 0.70 \
    --min-syntax-rate 0.0 \
    --no-require-delimiters \
    --eval-sample-size 100 \
    --eval-seed 123 \
    --eval-max-steps 400 \
    --eval-step-token-budget 8 \
    --vllm-tensor-parallel-size 1 \
    --vllm-max-model-len 4096 \
    --vllm-gpu-memory-utilization 0.40 \
    --synthesis-max-tokens 2048
