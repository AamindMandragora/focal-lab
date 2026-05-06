#!/bin/bash
# Generate a Spider-specific CSD for text-to-SQL reasoning.
#
# Usage: bash scripts/generate_sql_csd.sh

set -e

# SQL/Spider task description - aligned with the evaluator output contract.
TASK_DESC="Generate a Spider SQL query that answers the natural-language question using only the provided database schema. \
The answer contract is one SQL query only: no explanations, no code fences, and no text after the query. \
Keep the SQL query inside the hidden constrained parser-guided chunk."

echo "Generating Spider-specific CSD for SQL windows..."
echo ""
echo "Task description:"
echo "  $TASK_DESC"
echo ""

# Run synthesis. Thresholds (--min-accuracy, --min-syntax-rate) gate strategy
# acceptance in the feedback loop; delimiter-containment is required by default
# via --require-delimiters (already default-true in run_synthesis.py).
python run_synthesis.py \
    --task "$TASK_DESC" \
    --dataset spider \
    --max-iterations 20 \
    --generation-model "gpt-5.4" \
    --generation-backend openai \
    --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --eval-backend vllm \
    --output-name "sql_crane_csd" \
    --temperature 0.7 \
    --device auto \
    --min-accuracy 0.68 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 30 \
    --eval-seed 123 \
    --eval-max-steps 200 \
    --eval-step-token-budget 8 \
    --vllm-max-model-len 8192 \
    --vllm-tensor-parallel-size 1 \
    --vllm-gpu-memory-utilization 0.5 \
    --synthesis-max-tokens 2048

echo ""
echo "SQL CSD generation complete!"
echo ""
echo "To use the generated CSD, run:"
echo "CUDA_VISIBLE_DEVICES=0 python -m evaluations.sql_spider.cli \\"
echo "   --run-dir \$(cat outputs/generated-csd/latest_run.txt) \\"
echo "   --model Qwen/Qwen2.5-Coder-7B-Instruct \\"
echo "   --device cuda \\"
echo "   --limit 50 \\"
echo "   --max-steps 400 \\"
echo "   --load-in-4bit"
