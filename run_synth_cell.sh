#!/usr/bin/env bash
# Parameterized Qwen3.5 CSD synthesis launcher. COLD (no --initial-strategy-file, ever).
# REQUIRES the csd conda env (transformers 5.5.4 + vllm 0.19.1) for Qwen3.5 model code.
# Author = Sonnet-4.6 Bedrock, thinking high. Adaptive helper mask ON + bandit. Billed.
set -uo pipefail

# ---- params (env-var driven) ----
DATASET="${DATASET:?set DATASET=gsm_symbolic|spider|smiles}"
EVAL_MODEL="${EVAL_MODEL:?set EVAL_MODEL=Qwen/Qwen3.5-2B|4B|9B}"
GPU="${GPU:?set GPU=index}"
MAX_ITERS="${MAX_ITERS:-40}"
SAMPLE_SIZE="${SAMPLE_SIZE:-49}"
MIN_ACC="${MIN_ACC:-0.0}"
MIN_SYN="${MIN_SYN:-0.0}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-900}"
# 600s = non-binding anti-hang guard, NOT a compute cap. CRANE imposes no wall-clock
# limit on generation (only a 600-token budget); a tight clock would unfairly cut off
# our heavier per-token re-sampling. Token budget (EVAL_MAX_STEPS) is the real limit.
EVAL_MAX_SECONDS="${EVAL_MAX_SECONDS:-600}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.40}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
OUTPUT_NAME="${OUTPUT_NAME:?set OUTPUT_NAME}"
SMILES_CLASS="${SMILES_CLASS:-acrylates}"
SPLIT_NAME="${SPLIT_NAME:-train}"
SPLIT_FILE="${SPLIT_FILE:-}"   # optional override; else per-dataset default below
# Author thinking mode (enabled|off) — off is ONLY for the thinking-ablation arm.
ANTHROPIC_THINKING="${ANTHROPIC_THINKING:-enabled}"

REPO="${REPO:-/home/aadivyar/csd-generation}"
PY="${PY:-/apps/conda/aadivyar/envs/csd/bin/python}"
export LD_LIBRARY_PATH=/apps/conda/aadivyar/envs/csd/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME=/home/aadivyar/.cache/huggingface
export TRANSFORMERS_CACHE=/home/aadivyar/.cache/huggingface
# Bedrock creds (AWS_BEARER_TOKEN_BEDROCK + AWS_REGION) — account 887730490125
set -a; source "$REPO/.env"; set +a

cd "$REPO"
OUT="outputs/generated/$OUTPUT_NAME"
mkdir -p "$OUT"

COMMON=(
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock
  --eval-model "$EVAL_MODEL" --eval-backend vllm
  --max-iterations "$MAX_ITERS"
  --output-name "$OUTPUT_NAME" --output-dir "$OUT"
  --min-accuracy "$MIN_ACC" --min-syntax-rate "$MIN_SYN"
  --eval-sample-size "$SAMPLE_SIZE" --eval-max-steps "$EVAL_MAX_STEPS" --eval-step-token-budget 1
  --eval-max-seconds-per-example "$EVAL_MAX_SECONDS" --eval-min-examples-before-threshold-stop "$SAMPLE_SIZE"
  --max-tokens 32768 --restart-after-stuck-iters 0
  --vllm-gpu-memory-utilization "$GPU_MEM_UTIL" --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" --device auto
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2
  --anthropic-thinking "$ANTHROPIC_THINKING" --anthropic-effort high --anthropic-thinking-display summarized
  --vllm-tensor-parallel-size 1
)

case "$DATASET" in
  gsm_symbolic)
    SPLIT="${SPLIT_FILE:-/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json}"
    TASK='Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.'
    EXTRA=(--dataset gsm_symbolic --gsm-split-file "$SPLIT" --gsm-split-name "$SPLIT_NAME")
    ;;
  spider)
    export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
    export CSD_API_MAX_RETRIES=10
    export CSD_RECURRENCE_PENALTY=0.3
    SPLIT="${SPLIT_FILE:-/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json}"
    TASK='Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.'
    EXTRA=(--dataset spider --spider-split-file "$SPLIT" --spider-split-name "$SPLIT_NAME")
    ;;
  smiles)
    # TASK string to be filled from pilot_smiles_uv.sh before any SMILES run.
    TASK="${SMILES_TASK:?set SMILES_TASK for smiles runs}"
    EXTRA=(--dataset smiles --smiles-classes "$SMILES_CLASS" --smiles-samples-per-class "$SAMPLE_SIZE")
    ;;
  *)
    echo "unknown DATASET=$DATASET"; exit 2;;
esac

echo "SYNTH_START dataset=$DATASET model=$EVAL_MODEL gpu=$GPU iters=$MAX_ITERS sample=$SAMPLE_SIZE bar=$MIN_ACC/$MIN_SYN split=$SPLIT_NAME out=$OUT $(date)"
$PY -m synthesis.run_synthesis --task "$TASK" "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee "$OUT/run.log"
ec=${PIPESTATUS[0]}
echo "SYNTH_DONE exit=$ec out=$OUT $(date)"
echo "SYNTH_${OUTPUT_NAME}_SENTINEL_DONE exit=$ec"
exit "$ec"
