#!/bin/bash
# Run IterGen on the 300 seed334 test examples using the ~/itergen native harness.
# v4: uses ~/itergen/case_studies/sql/eval_sql_seed334_test300.py — the same code
#     path that produced the 62%/92% first-100 result.
# DFA mask store for Qwen2TokenizerFast is cached in ~/itergen/cache/mask_stores.
set -eu

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
export SPLIT_FILE=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
export SPIDER_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/aadivyar/itergen:/home/aadivyar/itergen/itergen/syncode:/home/aadivyar/itergen/itergen/syncode/syncode:${PYTHONPATH:-}"

LOG=/tmp/itergen_spider7b_seed334_test300_v4_20260605.log

cd /home/aadivyar/itergen

echo "Running IterGen seed334-test300 from $(pwd)"
echo "Log: $LOG"

nohup /opt/anaconda/bin/python case_studies/sql/eval_sql_seed334_test300.py \
    > "$LOG" 2>&1 &

echo "IterGen seed334-test300 v4 pid: $!  log=$LOG"
