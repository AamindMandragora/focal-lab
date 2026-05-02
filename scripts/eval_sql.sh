#!/bin/bash
# Generic SQL Spider eval wrapper with first-N vs random-sample toggle.
#
# Usage:
#   bash eval_first_or_random.sh <STRATEGY_DIR> [options]
#
# Required:
#   STRATEGY_DIR   First positional arg, OR env var. Path to a compiled-strategy
#                  directory containing GeneratedCSD.py.
#
# Options (all env-overrideable):
#   LIMIT          # examples to evaluate. Default: 100
#   MAX_STEPS      Per-example step cap. Default: 400
#   MODEL          HF model id. Default: Qwen/Qwen2.5-Coder-14B-Instruct
#   GPU            CUDA_VISIBLE_DEVICES. Default: 0
#   GPU_MEM_UTIL   --vllm-gpu-memory-utilization. Default: 0.75
#   MAX_MODEL_LEN  --vllm-max-model-len. Default: 4096
#   RANDOM_SAMPLE  Use random.Random(SEED).sample for example selection.
#                  Set to 1 to enable; default 0 (first-N deterministic).
#   SEED           Random seed used when RANDOM_SAMPLE=1. Default: 123
#   PRED_DUMP      Optional path to dump full predictions+per-row results JSON.
#
# Examples:
#   bash eval_first_or_random.sh /path/to/strategy                # first-100
#   RANDOM_SAMPLE=1 SEED=127 bash eval_first_or_random.sh /path   # random sample
#   LIMIT=200 bash eval_first_or_random.sh /path                  # first-200

set -e
STRATEGY_DIR=${1:-${STRATEGY_DIR:?STRATEGY_DIR not set; pass as first arg or env var}}

LIMIT=${LIMIT:-100}
MAX_STEPS=${MAX_STEPS:-400}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}
GPU=${GPU:-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
RANDOM_SAMPLE=${RANDOM_SAMPLE:-0}
SEED=${SEED:-123}
PRED_DUMP=${PRED_DUMP:-}

cd ~/csd-generation
source ~/.profile
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Build the optional flags conditionally
RANDOM_FLAG=""
if [ "$RANDOM_SAMPLE" = "1" ]; then
    RANDOM_FLAG="--random-sample --seed $SEED"
fi
DUMP_ENV=""
if [ -n "$PRED_DUMP" ]; then
    DUMP_ENV="SQL_PRED_DUMP=$PRED_DUMP"
fi

echo "=== eval config ==="
echo "  strategy_dir   : $STRATEGY_DIR"
echo "  model          : $MODEL"
echo "  limit          : $LIMIT"
echo "  max-steps      : $MAX_STEPS"
echo "  GPU            : $GPU"
echo "  random_sample  : $RANDOM_SAMPLE${RANDOM_SAMPLE:+ (seed=$SEED)}"
echo "  pred_dump      : ${PRED_DUMP:-<disabled>}"
echo "==================="

env CUDA_VISIBLE_DEVICES=$GPU $DUMP_ENV /opt/anaconda/bin/python -m evaluations.sql_spider.cli \
    --run-dir "$STRATEGY_DIR" \
    --model "$MODEL" \
    --backend vllm \
    --device cuda \
    --limit $LIMIT \
    --max-steps $MAX_STEPS \
    --vllm-tensor-parallel-size 1 \
    --vllm-max-model-len $MAX_MODEL_LEN \
    --vllm-gpu-memory-utilization $GPU_MEM_UTIL \
    $RANDOM_FLAG
