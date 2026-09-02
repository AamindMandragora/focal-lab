#!/usr/bin/env bash
# ==============================================================================
# PURE RE-EVAL — Spider 9B (Qwen3.5-9B) att20 strategy on HELD-OUT test300
# ==============================================================================
#
# WHY: synth_spider_9b_seed334train300_0628 ACCEPTED at attempt 20 with
#   73.7% acc / 98.7% syn on TRAIN300 (crosses IterGen 67% bar). Per the
#   train-win-then-held-out rule, re-eval that accepted strategy on the
#   held-out test300 split — the number that lands in the matrix.
#
# SPLIT-NAME: --spider-split-name test  (resolves to test_indices, the 300
#   examples IterGen scored on; train_indices is what synthesis ran on).
#
# ACCEPTED ATT20 STRATEGY (full .dfy, run-dir root = accepted/best):
#   /home/aadivyar/csd-generation/outputs/generated/synth_spider_9b_seed334train300_0628/
#     synth_spider_9b_seed334train300_0628_20260628_024133_baa1d7/dafny/GeneratedCSD.dfy
#   Evidence: run.log "SUCCESS after 20 attempt(s)" + success_report.json
#   evaluation accuracy 0.737 / syntax 0.987, total_attempts 20.
#
# EVAL FLAGS — matched to the 9B train run:
#   max_steps=200 (IterGen max_new_tokens parity, fair)
#   gpu_memory_utilization=0.85 (9B weights need it; at 0.5 KV cache is too small
#   and the engine fails to initialize. Util only sizes KV cache, not semantics.)
#   recurrence_penalty=0.3 (CSD_RECURRENCE_PENALTY, as in the train run)
#
# COLD-RULE COMPLIANCE: --initial-strategy-file + --max-iterations 1 + bars 0.
#   Pure re-eval, not synthesis. No author/generation call fires.
#
# USAGE: bash fire_spider9b_heldout_reeval.sh <GPU_INDEX>
# ==============================================================================

set -euo pipefail

GPU="${1:?Usage: $0 <GPU_INDEX>}"

REPO=/home/aadivyar/csd-generation
PY=/apps/conda/aadivyar/envs/csd/bin/python
FULL_DFY="$REPO/outputs/generated/synth_spider_9b_seed334train300_0628/synth_spider_9b_seed334train300_0628_20260628_024133_baa1d7/dafny/GeneratedCSD.dfy"
STRATEGY_BODY=/tmp/spider9b_att20_strategy_body.dfy
SPLIT_FILE="$REPO/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"
OUTPUT_NAME="reeval_spider_9b_att20_HELDOUT_test300_0628"
OUT="$REPO/outputs/generated/$OUTPUT_NAME"
LAUNCH_LOG="$OUT/launch.log"

# ---- Extract strategy body from full .dfy ----
echo "Extracting strategy body from full .dfy..."
awk '/\/\/ CSD_RATIONALE_BEGIN/{found=1} found{print}' "$FULL_DFY" | head -n -2 > "$STRATEGY_BODY"

FIRST_LINE=$(head -1 "$STRATEGY_BODY")
if [[ "$FIRST_LINE" != *"CSD_RATIONALE_BEGIN"* ]]; then
    echo "ERROR: Extracted body does not start with CSD_RATIONALE_BEGIN (got: $FIRST_LINE)"
    exit 1
fi
BODY_LINES=$(wc -l < "$STRATEGY_BODY")
echo "Strategy body: $STRATEGY_BODY ($BODY_LINES lines, first line: $FIRST_LINE)"

mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=/apps/conda/aadivyar/envs/csd/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/home/aadivyar/.cache/huggingface
export TRANSFORMERS_CACHE=/home/aadivyar/.cache/huggingface
export SPIDER_DB_DIR="$REPO/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases"
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3

set -a; source "$REPO/.env"; set +a

echo "REEVAL_START output=$OUTPUT_NAME gpu=$GPU strategy=$STRATEGY_BODY split=test/test_indices $(date)"

nohup "$PY" -m synthesis.run_synthesis \
    --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
    --dataset spider \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "Qwen/Qwen3.5-9B" \
    --eval-backend vllm \
    --initial-strategy-file "$STRATEGY_BODY" \
    --max-iterations 1 \
    --output-name "$OUTPUT_NAME" \
    --output-dir "$OUT" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 300 \
    --eval-max-steps 200 \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 600 \
    --eval-min-examples-before-threshold-stop 300 \
    --vllm-gpu-memory-utilization 0.85 \
    --device auto \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --refinement-beam-size 2 \
    --restart-after-stuck-iters 0 \
    --max-tokens 32768 \
    --vllm-tensor-parallel-size 1 \
    --spider-split-file "$SPLIT_FILE" \
    --spider-split-name test \
    >> "$LAUNCH_LOG" 2>&1

EC=$?
echo "REEVAL_DONE exit=$EC out=$OUT $(date)" | tee -a "$LAUNCH_LOG"
echo "REEVAL_${OUTPUT_NAME}_SENTINEL_DONE exit=$EC"
