#!/bin/bash
# Run IterGen baseline on the same 300 seed334 test examples used in our held-out eval.
# Purpose: fair apples-to-apples comparison vs our 57.3%/97% result.
# v2: adds --enable_dist True to route through process_dataset_parallel (Spider IterGen
#     requires this path; process_dataset_interactive does not support spider+itergen).
# DFA mask store (560 items) was built by v1 and is now cached — this run skips that 43m build.
set -eu

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

SPLIT_FILE=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
LOG=/tmp/itergen_spider7b_seed334_test300_v2_20260605.log
CRANE_SRC=/home/aadivyar/CRANE/src

# Extract the 300 seed334 test indices as a comma-separated string
INDICES=$(python3 -c "
import json
d = json.load(open('$SPLIT_FILE'))
print(','.join(str(i) for i in sorted(d['test_indices'])))
")

echo "Running IterGen on $(echo $INDICES | tr ',' '\n' | wc -l) seed334 test examples"
echo "Indices sample: $(echo $INDICES | cut -d',' -f1-5)..."

cd "$CRANE_SRC"

nohup /apps/conda/advayth2/envs/advayth2/bin/python main.py \
    --dataset spider \
    --indices "$INDICES" \
    --num_shots 8 \
    --overwrite_results True \
    --write_file True \
    --regex_parser True \
    --modify_system_prompt True \
    --cot_model "Qwen/Qwen2.5-7B-Instruct" \
    --cot_grammar_mode itergen \
    --cot_grammar sql \
    --out_grammar sql \
    --max_tokens 1200 \
    --temperature 0.0 \
    --start_symbol "<<" \
    --end_symbol ">>" \
    --do_cot True \
    --enable_dist True \
    --cot_device cuda:0 \
    --llm_parser_device cuda:0 \
    > "$LOG" 2>&1 &

echo "IterGen seed334-test300 v2 pid: $!  log=$LOG"
