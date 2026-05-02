#!/bin/bash
# Generate a Spider-specific CSD for text-to-SQL reasoning.
#
# Usage: bash scripts/generate_sql_csd.sh

set -e

# SQL/Spider task description - bare description, no strategy hints.
TASK_DESC="Text-to-SQL generation on the Spider benchmark. The model reads a schema (tables and columns) and \
a natural-language question, then emits a single SQL query as its output. The parser validates the query \
against a SQL grammar that is dynamically narrowed to the current schema's tables and columns."

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
