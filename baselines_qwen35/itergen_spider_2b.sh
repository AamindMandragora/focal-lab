#!/usr/bin/env bash
# IterGen Spider baseline: Qwen/Qwen3.5-2B on the seed334 test-300 split.
#
# Mechanism: eval_sql_seed334_test300.py (native IterGen harness) with SPIDER_MODEL_ID
#   and SPLIT_FILE env vars. Uses max_new_tokens=200 (hardcoded in the script to match
#   IterGen's published max_new_tokens, per the fairness fix from 2026-06-23).
#
# $0 cost: no Bedrock; model loaded locally via HuggingFace in the IterGen env.
#
# PREVIOUSLY RUN: 2026-06-26 via run_qwen35_spider_baselines.sh.
#   Raw result file: ~/itergen/results/sql_results/Qwen/Qwen3.5-2B/itergen_seed334_test300_tempt:None_seed:0_rp:0.3_maxiter_20_num:300.jsonl
#   Rescored (seed334 gold alignment fix): 113/300 = 37.7% exec accuracy
#   Syntax rate: Valid=221, Syntax Error=28 -> 90.7% (272/300)
#
# Usage: GPU=N bash itergen_spider_2b.sh  (default GPU=0)

set -u
cd /home/aadivyar/itergen

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE="cuda:0"
export SPIDER_MODEL_ID="Qwen/Qwen3.5-2B"
export SPLIT_FILE="/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"

PY=/apps/conda/aadivyar/envs/csd/bin/python
LOGDIR=/home/aadivyar/csd-generation/outputs/generated/baseline_itergen_spider_2b_heldout_0627
LOG="$LOGDIR/run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOGDIR"
echo "BASELINE_START itergen_spider_2b model=$SPIDER_MODEL_ID gpu=$GPU $(date -u)" | tee "$LOG"

"$PY" case_studies/sql/eval_sql_seed334_test300.py >> "$LOG" 2>&1

EXIT=$?
echo "SENTINEL_DONE itergen_spider_2b exit=$EXIT $(date -u)" | tee -a "$LOG"

# Print rescored accuracy automatically after the run
echo "--- Rescoring with seed334 gold alignment fix ---" | tee -a "$LOG"
RESULTS_JSONL="results/sql_results/Qwen/Qwen3.5-2B/itergen_seed334_test300_tempt:None_seed:0_rp:0.3_maxiter_20_num:300.jsonl" \
    "$PY" case_studies/sql/rescore_itergen_seed334.py 2>&1 | tee -a "$LOG"
