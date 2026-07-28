#!/usr/bin/env bash
# ==============================================================================
# PURE RE-EVAL — Spider 2B (Qwen3.5-2B) att17 strategy on HELD-OUT test300
# ==============================================================================
#
# SPLIT-NAME DECISION: --spider-split-name test
#   evaluator.py:1706-1710 (_normalize_spider_split_name): "eval" → "test";
#   "test" stays "test". Both load key "test_indices" from the manifest.
#   spider_dev_proportional_300x300_seed334.json has:
#     test_indices: 300 examples  ← the held-out set used here
#     train_indices: 300 examples ← what the synthesis ran on
#     eval_indices: 300 examples  ← a THIRD set; NOT used here
#   The existing 1.5B held-out reeval (launch_spider1p5b_300x300_seed334_heldout_reeval_20260604.sh)
#   used --spider-split-name eval, which also resolves to test_indices via the normalizer.
#   We use "test" directly for clarity.
#
# ACCEPTED ATT17 STRATEGY:
#   Full .dfy: /home/aadivyar/csd-generation/outputs/generated/
#     synth_spider_2b_seed334train300_0627c/
#     synth_spider_2b_seed334train300_0627c_20260627_133305_7ab948/dafny/GeneratedCSD.dfy
#   Evidence: run.log tail — "Total attempts: 17 / Accuracy: 39.0% / Syntax: 99.0%"
#   The _092442_3a3f1e subdir also exists but its run.log shows an intermediate run;
#   _133305_7ab948 is the one that completed att17.
#
# EVAL FLAGS — matched to the original train run:
#   max_steps=200 (confirmed: every "[EVAL] Running CSD strategy (max_steps=200)" in run.log)
#   gpu_memory_utilization=0.5 (confirmed: run.log "non-default args: gpu_memory_utilization: 0.5")
#
# COLD-RULE COMPLIANCE (per project CLAUDE.md warm-starts-banned rule):
#   --initial-strategy-file is the ONLY legitimate use when paired with
#   --max-iterations 1 and bars 0. This is a pure re-eval, not synthesis.
#   No author/generation call fires. No information is fed back.
#
# USAGE: bash fire_spider2b_heldout_reeval.sh <GPU_INDEX>
# ==============================================================================

set -euo pipefail

GPU="${1:?Usage: $0 <GPU_INDEX>}"

REPO=/home/aadivyar/csd-generation
PY=/apps/conda/aadivyar/envs/csd/bin/python
FULL_DFY="$REPO/outputs/generated/synth_spider_2b_seed334train300_0627c/synth_spider_2b_seed334train300_0627c_20260627_133305_7ab948/dafny/GeneratedCSD.dfy"
STRATEGY_BODY=/tmp/spider2b_att17_strategy_body.dfy
SPLIT_FILE="$REPO/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"
OUTPUT_NAME="reeval_spider_2b_att17_HELDOUT_test300_0628"
OUT="$REPO/outputs/generated/$OUTPUT_NAME"
LAUNCH_LOG="$OUT/launch.log"

# ---- Extract strategy body from full .dfy ----
# GeneratedCSD.dfy is the full class file (module wrapper + template).
# The body for --initial-strategy-file must start with // CSD_RATIONALE_BEGIN
# and have no class wrapper. Extraction: grab from CSD_RATIONALE_BEGIN to end,
# then drop the last 2 lines (the two closing `}` that close AuthorBody + module).
echo "Extracting strategy body from full .dfy..."
awk '/\/\/ CSD_RATIONALE_BEGIN/{found=1} found{print}' "$FULL_DFY" | head -n -2 > "$STRATEGY_BODY"

# Verify the body starts correctly and is non-empty
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
    --eval-model "Qwen/Qwen3.5-2B" \
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
    --eval-max-seconds-per-example 90 \
    --eval-min-examples-before-threshold-stop 300 \
    --vllm-gpu-memory-utilization 0.5 \
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
