#!/bin/bash
# Re-run of the Spider-1.5B 300x300 seed334 held-out eval, now scoring against the
# CORRECT test-suite ("fixed") databases the baselines use. The DB-dir root cause
# (proven 2026-06-06) was fixed in dataset.py default_db_dir(); we ALSO set
# SPIDER_DB_DIR here explicitly as a belt-and-suspenders guarantee for this run.
# Expected pipeline accuracy ~51% (matching the read-only official rescore), vs the
# old 39% under the raw databases. Same strategy + same split + same eval config as
# the original spider1p5b_300x300_seed334_heldout_reeval_20260604 run.
set -eu

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases

cd /home/aadivyar/csd-generation

STRATEGY=/home/aadivyar/csd-generation/spider1p5b_300x300_warmstart_body.dfy
echo "Strategy: $STRATEGY"
echo "SPIDER_DB_DIR: $SPIDER_DB_DIR"

SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
OUT=outputs/generated/spider1p5b_300x300_seed334_heldout_dbfix_20260606
mkdir -p "$OUT"

/apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
    --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
    --dataset spider \
    --max-iterations 1 \
    --initial-strategy-file "$STRATEGY" \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --eval-backend vllm \
    --output-name "spider1p5b_300x300_seed334_heldout_dbfix_20260606" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 300 \
    --eval-max-steps 1200 \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 180 \
    --eval-min-examples-before-threshold-stop 300 \
    --vllm-gpu-memory-utilization 0.20 \
    --device auto \
    --output-dir "$OUT" \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --anthropic-thinking enabled \
    --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --vllm-tensor-parallel-size 1 \
    --spider-split-file "$SPIDER_SPLIT" \
    --spider-split-name eval \
    2>&1 | tee /tmp/spider1p5b_300x300_seed334_heldout_dbfix_20260606.log
echo "Spider-1.5B 300x300 held-out reeval (DB fix) done -> $OUT"
