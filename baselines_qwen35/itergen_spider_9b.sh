#!/usr/bin/env bash
# IterGen Spider baseline: Qwen/Qwen3.5-9B on the seed334 test-300 split.
#
# Mechanism: eval_sql_seed334_test300.py with max_new_tokens=200 (IterGen-fair).
# $0 cost: no Bedrock; local HuggingFace model loading.
# 9B model: needs a full GPU.
#
# PREVIOUSLY RUN: 2026-06-26.
#   Rescored: 201/300 = 67.0% exec accuracy
#   Syntax rate: Valid=278, Syntax Error=5 -> 98.3% (295/300)
#
# Usage: GPU=N bash itergen_spider_9b.sh  (default GPU=0)

set -u
cd /home/aadivyar/itergen

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE="cuda:0"
export SPIDER_MODEL_ID="Qwen/Qwen3.5-9B"
export SPLIT_FILE="/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"

PY=/apps/conda/aadivyar/envs/csd/bin/python
LOGDIR=/home/aadivyar/csd-generation/outputs/generated/baseline_itergen_spider_9b_heldout_0627
LOG="$LOGDIR/run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOGDIR"
echo "BASELINE_START itergen_spider_9b model=$SPIDER_MODEL_ID gpu=$GPU $(date -u)" | tee "$LOG"

"$PY" case_studies/sql/eval_sql_seed334_test300.py >> "$LOG" 2>&1

EXIT=$?
echo "SENTINEL_DONE itergen_spider_9b exit=$EXIT $(date -u)" | tee -a "$LOG"

echo "--- Rescoring with seed334 gold alignment fix ---" | tee -a "$LOG"
RESULTS_JSONL="results/sql_results/Qwen/Qwen3.5-9B/itergen_seed334_test300_tempt:None_seed:0_rp:0.3_maxiter_20_num:300.jsonl" \
    "$PY" case_studies/sql/rescore_itergen_seed334.py 2>&1 | tee -a "$LOG"
