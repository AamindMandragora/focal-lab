#!/bin/bash
echo "Hello, FOCAL!"
echo "Beginning testing!"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_synthesis.py \
  --dataset datasets/ml-gsm-symbolic/generated_data/GSM_symbolic.jsonl \
  --compiled-module outputs/generated-csd/runs/20260106_192252_5ed275/generated_csd \
  --parser-mode math \
  --lark-file grammars/math.lark \
  --model Qwen/Qwen2.5-Coder-3B-Instruct
echo "I can't believe that worked..."