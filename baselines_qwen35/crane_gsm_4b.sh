#!/usr/bin/env bash
# CRANE baseline re-eval: Qwen/Qwen3.5-4B on GSM-Symbolic seed123 eval-49.
#
# Mechanism: synthesis.evaluate.run_legacy_fixed_strategy --strategy crane
#   -> crane_repo_runner -> CRANE main.py --cot_grammar_mode adaptive --do_cot True
#
# $0 cost: no Bedrock; CRANE uses HuggingFace model loading in its subprocess.
#
# Usage: GPU=N bash crane_gsm_4b.sh  (default GPU=0)
#
# PREVIOUSLY RUN: 2026-06-26. Recorded result: 21/49 = 42.9% accuracy.
#   Re-run only if you need the standard baseline JSON format.

set -u
cd /home/aadivyar/csd-generation

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}

PY=/apps/conda/aadivyar/envs/csd/bin/python
SPLIT=environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
OUTDIR=outputs/baselines/crane/Qwen_Qwen3_5-4B
OUTFILE="$OUTDIR/gsm_symbolic__tb1__ms900.json"
LOG=logs/crane_gsm_4b_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$OUTDIR" logs
echo "BASELINE_START crane_gsm_4b model=Qwen/Qwen3.5-4B gpu=$GPU $(date -u)" | tee "$LOG"

"$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
    --strategy crane \
    --dataset gsm_symbolic \
    --eval-model Qwen/Qwen3.5-4B \
    --eval-backend huggingface \
    --eval-sample-size 49 \
    --eval-max-steps 900 \
    --eval-step-token-budget 1 \
    --gsm-split-file "$SPLIT" \
    --gsm-split-name eval \
    --output-json "$OUTFILE" \
    >> "$LOG" 2>&1

EXIT=$?
echo "SENTINEL_DONE crane_gsm_4b exit=$EXIT $(date -u)" | tee -a "$LOG"
