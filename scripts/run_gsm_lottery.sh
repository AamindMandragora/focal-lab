#!/bin/bash
export PATH=/opt/anaconda/bin:$PATH
cd ~/csd-generation
for SEED in 200 300 400 500 600 700 800 900 1000 1100; do
    echo "=== Lottery seed base=$SEED $(date) ==="
    CUDA_VISIBLE_DEVICES=1 python run_synthesis.py \
        --task 'Solve GSM-Symbolic math word problems. Natural-language reasoning may stay outside constrained spans; for scoring, the required constrained region is the final answer only. The last <<...>> span must contain one valid symbolic arithmetic expression that answers the question. Do not require every intermediate calculation to be wrapped.' \
        --dataset gsm_symbolic \
        --max-iterations 12 \
        --generation-model gpt-5.4 \
        --generation-backend openai \
        --eval-model Qwen/Qwen2.5-Coder-14B-Instruct \
        --eval-backend vllm \
        --output-name gsm_lottery_seed${SEED} \
        --temperature 0.7 \
        --device auto \
        --min-accuracy 0.51 \
        --min-syntax-rate 0.72 \
        --eval-sample-size 10 \
        --eval-seed ${SEED} \
        --eval-max-steps 512 \
        --eval-step-token-budget 1 \
        --vllm-max-model-len 8192 \
        --vllm-tensor-parallel-size 1 \
        --vllm-gpu-memory-utilization 0.85 \
        --synthesis-max-tokens 2048
    echo "=== Seed ${SEED} done ==="
done
echo "=== Full lottery complete ==="
