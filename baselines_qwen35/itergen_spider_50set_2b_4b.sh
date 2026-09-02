#!/usr/bin/env bash
# ==============================================================================
# IterGen Spider baseline on the 50-EXAMPLE fast-iteration set (seed334).
#   Runs Qwen3.5-2B then Qwen3.5-4B sequentially on ONE GPU (co-located, safe:
#   IterGen uses HF transformers + DynamicCache, ~7GB/2B ~12GB/4B, no fixed-util
#   grab, so it won't OOM a colleague's resident model on the same card).
#
# WHY: the pivot needs the baseline's score on EXACTLY the same 50 examples we'll
#   fast-iterate metaDecode on, so the win bar is fair. $0 (no Bedrock; local HF).
#
# 50-set split: spider_dev_50eval_for_itergen_seed334.json — its 50 indices live
#   in test_indices, which is what eval_sql_seed334_test300.py reads (line 156).
#   IterGen settings are hardcoded in that script (rp=0.3, max_iter=20,
#   temperature=None, seed=0) and match the recorded 300-set bars exactly.
#
# Trustworthy number = rescore_itergen_seed334.py on the num:50 jsonl (the inline
#   print can hit the known gold-misalignment artifact).
#
# Usage: GPU=2 bash itergen_spider_50set_2b_4b.sh
# ==============================================================================
set -u
cd /home/aadivyar/itergen

GPU=${GPU:-2}
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE="cuda:0"
export SPLIT_FILE="/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_50eval_for_itergen_seed334.json"

PY=/apps/conda/aadivyar/envs/csd/bin/python
LOGDIR=/home/aadivyar/csd-generation/outputs/generated/baseline_itergen_spider_50set_0628
mkdir -p "$LOGDIR"
LOG="$LOGDIR/run_$(date +%Y%m%d_%H%M%S).log"

run_model () {
  local MODEL="$1"
  echo "=================================================================" | tee -a "$LOG"
  echo "BASELINE_START itergen_spider_50set model=$MODEL gpu=$GPU $(date -u)" | tee -a "$LOG"
  export SPIDER_MODEL_ID="$MODEL"
  "$PY" case_studies/sql/eval_sql_seed334_test300.py >> "$LOG" 2>&1
  local EXIT=$?
  echo "EVAL_DONE model=$MODEL exit=$EXIT $(date -u)" | tee -a "$LOG"

  # Rescore (seed334 gold alignment fix) on the num:50 jsonl this run wrote.
  local RJ="results/sql_results/${MODEL}/itergen_seed334_test300_tempt:None_seed:0_rp:0.3_maxiter_20_num:50.jsonl"
  echo "--- Rescoring $MODEL on 50-set ($RJ) ---" | tee -a "$LOG"
  if [[ -f "$RJ" ]]; then
    RESULTS_JSONL="$RJ" "$PY" case_studies/sql/rescore_itergen_seed334.py 2>&1 | tee -a "$LOG"
  else
    echo "ERROR: results jsonl not found: $RJ" | tee -a "$LOG"
  fi
  echo "RESCORE_DONE model=$MODEL $(date -u)" | tee -a "$LOG"
}

echo "LOG=$LOG"
run_model "Qwen/Qwen3.5-2B"
run_model "Qwen/Qwen3.5-4B"
echo "ALL_SENTINEL_DONE itergen_spider_50set 2b+4b $(date -u)" | tee -a "$LOG"
