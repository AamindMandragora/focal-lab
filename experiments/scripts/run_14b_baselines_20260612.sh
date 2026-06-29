#!/bin/bash
# Master sequential baseline sweep for Qwen2.5-14B-Instruct — GSM-Symbolic + Spider.
# Reproduces exactly the controlled-comparison rows that exist for 1.5B/7B.
#
# Inputs:
#   $GPU  — set by waiter_gpu_14b_baselines.sh (the single GPU index that has >= 34 GB free)
#   All split files, CRANE repo, itergen repo, and cars library must already be present.
#
# Outputs:
#   GSM : outputs/controlled_comparison/gsm_14B/{unconstrained,gcd,crane,itergen,cars}.json
#   Spider (run_legacy_fixed_strategy):
#         outputs/baselines/cars/Qwen_Qwen2.5_14B_Instruct/spider_seed334_test300__tb1__ms600.json
#   Spider (itergen-native):
#         ~/itergen/results/sql_results/Qwen/Qwen2.5-14B-Instruct/
#             unconstrained_native_seed334_test300_tempt:None_seed:0_num:300.jsonl
#             itergen_seed334_test300_tempt:None_seed:0_rp:0.3_maxiter_20_num:300.jsonl
#
# Algorithm (sequential to avoid OOM; 14B weights ~28 GB on a 40 GB card):
#   1. GSM unconstrained, gcd, crane, itergen — via run_legacy_fixed_strategy + CRANE pipeline
#      (huggingface backend so CRANE repo's forked grader handles scoring natively)
#   2. GSM cars — same harness, CARS upstream library, cpu-side decode loop
#   3. Spider unconstrained — itergen-native harness (eval_sql_unconstrained_seed334_test300.py)
#   4. Spider itergen — itergen-native harness (eval_sql_seed334_test300.py)
#   5. Spider cars — run_legacy_fixed_strategy --strategy cars --dataset spider
#   6. Spider gcd — run_legacy_fixed_strategy --strategy gcd --dataset spider (vllm)
#
# Note: GPU memory utilisation for vLLM is 0.85 (~34 GB), which fits comfortably on a
# newly-freed 40 GB card.  The huggingface-backend GSM steps load the model via
# Transformers; their peak is similar (~28 GB weights + KV cache).
#
# DO NOT MODIFY grammars, graders, split files, or existing output files.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

set -uo pipefail

# ── sanity check ────────────────────────────────────────────────────────────────
: "${GPU:?ERROR: GPU env var must be set by the caller (waiter script)}"

MODEL=Qwen/Qwen2.5-14B-Instruct
M14_TAG=Qwen_Qwen2.5_14B_Instruct       # directory-name form

GSPLIT="$SPLITS_DIR/gsm_symbolic_crane_proportional_49x49_seed123.json"
SSPLIT="$SPLITS_DIR/spider_dev_proportional_300x300_seed334.json"

OUT_GSM=$ROOT/outputs/controlled_comparison/gsm_14B
OUT_SPIDER_CARS=$ROOT/outputs/baselines/cars/${M14_TAG}

MASTERLOG=$ROOT/run_14b_baselines_20260612.log
# PY from lib.sh
mkdir -p "$OUT_GSM" "$OUT_SPIDER_CARS"
mkdir -p "$ROOT/logs/baselines_14B"

export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTERLOG"; }

log "=== START 14B BASELINES GPU=$GPU $(date) ==="
log "MODEL=$MODEL  GSM_SPLIT=$GSPLIT  SPIDER_SPLIT=$SSPLIT"

# ── helper: skip if output already exists and is non-empty ───────────────────
skip_if_exists() {
    local outfile="$1" label="$2"
    if [ -s "$outfile" ]; then
        log "SKIP $label (output already exists: $outfile)"
        return 0
    fi
    return 1
}

# ────────────────────────────────────────────────────────────────────────────────
# SECTION A — GSM-Symbolic controlled seed123 eval-49
# Source convention:  crane_repo controlled  (same as 1.5B/7B rows)
# All four CRANE-repo strategies run through run_legacy_fixed_strategy
# with --eval-backend huggingface (matches gsm7b_baselines_rerun.sh exactly).
# CARS uses the upstream ~/cars library (no --eval-backend flag needed; it
# ignores vllm and calls the library directly regardless of backend).
# ────────────────────────────────────────────────────────────────────────────────

log "=== 14B GSM unconstrained $(date) ==="
GSM_UNC_OUT="$OUT_GSM/unconstrained.json"
if ! skip_if_exists "$GSM_UNC_OUT" "gsm unconstrained"; then
    "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
        --strategy unconstrained --dataset gsm_symbolic \
        --eval-model "$MODEL" \
        --eval-backend huggingface \
        --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
        --gsm-split-file "$GSPLIT" --gsm-split-name eval \
        --output-json "$GSM_UNC_OUT" \
        >> "$ROOT/logs/baselines_14B/gsm_unconstrained_20260612.log" 2>&1
fi
EXIT_unconstrained_gsm14B=$?
log "EXIT_unconstrained_gsm14B=$EXIT_unconstrained_gsm14B"

log "=== 14B GSM gcd $(date) ==="
GSM_GCD_OUT="$OUT_GSM/gcd.json"
if ! skip_if_exists "$GSM_GCD_OUT" "gsm gcd"; then
    "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
        --strategy gcd --dataset gsm_symbolic \
        --eval-model "$MODEL" \
        --eval-backend huggingface \
        --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
        --gsm-split-file "$GSPLIT" --gsm-split-name eval \
        --output-json "$GSM_GCD_OUT" \
        >> "$ROOT/logs/baselines_14B/gsm_gcd_20260612.log" 2>&1
fi
EXIT_gcd_gsm14B=$?
log "EXIT_gcd_gsm14B=$EXIT_gcd_gsm14B"

log "=== 14B GSM crane $(date) ==="
GSM_CRANE_OUT="$OUT_GSM/crane.json"
if ! skip_if_exists "$GSM_CRANE_OUT" "gsm crane"; then
    "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
        --strategy crane --dataset gsm_symbolic \
        --eval-model "$MODEL" \
        --eval-backend huggingface \
        --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
        --gsm-split-file "$GSPLIT" --gsm-split-name eval \
        --output-json "$GSM_CRANE_OUT" \
        >> "$ROOT/logs/baselines_14B/gsm_crane_20260612.log" 2>&1
fi
EXIT_crane_gsm14B=$?
log "EXIT_crane_gsm14B=$EXIT_crane_gsm14B"

log "=== 14B GSM itergen $(date) ==="
GSM_ITER_OUT="$OUT_GSM/itergen.json"
if ! skip_if_exists "$GSM_ITER_OUT" "gsm itergen"; then
    "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
        --strategy itergen --dataset gsm_symbolic \
        --eval-model "$MODEL" \
        --eval-backend huggingface \
        --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
        --gsm-split-file "$GSPLIT" --gsm-split-name eval \
        --output-json "$GSM_ITER_OUT" \
        >> "$ROOT/logs/baselines_14B/gsm_itergen_20260612.log" 2>&1
fi
EXIT_itergen_gsm14B=$?
log "EXIT_itergen_gsm14B=$EXIT_itergen_gsm14B"

log "=== 14B GSM cars $(date) ==="
GSM_CARS_OUT="$OUT_GSM/cars.json"
if ! skip_if_exists "$GSM_CARS_OUT" "gsm cars"; then
    "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
        --strategy cars --dataset gsm_symbolic \
        --eval-model "$MODEL" \
        --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
        --cars-search-steps 200 \
        --gsm-split-file "$GSPLIT" --gsm-split-name eval \
        --output-json "$GSM_CARS_OUT" \
        >> "$ROOT/logs/baselines_14B/gsm_cars_20260612.log" 2>&1
fi
EXIT_cars_gsm14B=$?
log "EXIT_cars_gsm14B=$EXIT_cars_gsm14B"

log "=== GSM-14B RESULTS $(date) ==="
"$PY" - <<'PY' 2>/dev/null | tee -a "$MASTERLOG" || log "results print failed"
import json, glob, os
out = "$ROOT/outputs/controlled_comparison/gsm_14B"
for p in sorted(glob.glob(f"{out}/*.json")):
    try:
        d = json.load(open(p))
        name = os.path.basename(p)
        print(f"  {name}: acc={d.get('accuracy',0):.3f} syn={d.get('syntax_rate',0):.3f} n={len(d.get('answers',[]))}")
    except Exception as e:
        print(f"  {p}: ERROR {e}")
PY

# ────────────────────────────────────────────────────────────────────────────────
# SECTION B — Spider seed334 test-300
# Three routes mirror the 1.5B/7B approach exactly:
#   B1. Unconstrained — itergen-native harness (eval_sql_unconstrained_seed334_test300.py)
#   B2. IterGen       — itergen-native harness (eval_sql_seed334_test300.py)
#   B3. CARS          — run_legacy_fixed_strategy --strategy cars --dataset spider
#   B4. GCD/SynCode   — run_legacy_fixed_strategy --strategy gcd --dataset spider (vllm)
# B1/B2 use ITERGEN_NATIVE_PY (historically /opt/anaconda) when available.
# B3/B4 use the advayth2 conda python (same vllm env as all other CSD baselines).
# ────────────────────────────────────────────────────────────────────────────────

# B1 — Spider unconstrained (itergen-native)
log "=== 14B Spider unconstrained native $(date) ==="
UNC_JSONL="$ITERGEN_ROOT/results/sql_results/${MODEL}/unconstrained_native_seed334_test300_tempt:None_seed:0_num:300.jsonl"
if [ -s "$UNC_JSONL" ]; then
    log "SKIP spider unconstrained (output already exists: $UNC_JSONL)"
    EXIT_unconstrained_spider14B=0
else
    mkdir -p "$(dirname "$UNC_JSONL")"
    (
        cd $ITERGEN_ROOT
        export PYTHONPATH="$ITERGEN_ROOT:$ITERGEN_ROOT/itergen/syncode:$ITERGEN_ROOT/itergen/syncode/syncode:${PYTHONPATH:-}"
        export SPLIT_FILE="$SSPLIT"
        export SPIDER_MODEL_ID="$MODEL"
        export TOKENIZERS_PARALLELISM=false
        export CUDA_DEVICE="cuda:0"   # maps to $GPU since CUDA_VISIBLE_DEVICES is set
        "$ITERGEN_NATIVE_PY" case_studies/sql/eval_sql_unconstrained_seed334_test300.py
    ) >> "$ROOT/logs/baselines_14B/spider_unconstrained_20260612.log" 2>&1
    EXIT_unconstrained_spider14B=$?
fi
log "EXIT_unconstrained_spider14B=$EXIT_unconstrained_spider14B"

# B2 — Spider itergen (itergen-native)
log "=== 14B Spider itergen native $(date) ==="
ITER_JSONL="$ITERGEN_ROOT/results/sql_results/${MODEL}/itergen_seed334_test300_tempt:None_seed:0_rp:0.3_maxiter_20_num:300.jsonl"
if [ -s "$ITER_JSONL" ]; then
    log "SKIP spider itergen (output already exists: $ITER_JSONL)"
    EXIT_itergen_spider14B=0
else
    mkdir -p "$(dirname "$ITER_JSONL")"
    (
        cd $ITERGEN_ROOT
        export PYTHONPATH="$ITERGEN_ROOT:$ITERGEN_ROOT/itergen/syncode:$ITERGEN_ROOT/itergen/syncode/syncode:${PYTHONPATH:-}"
        export SPLIT_FILE="$SSPLIT"
        export SPIDER_MODEL_ID="$MODEL"
        export TOKENIZERS_PARALLELISM=false
        export CUDA_DEVICE="cuda:0"   # maps to $GPU since CUDA_VISIBLE_DEVICES is set
        "$ITERGEN_NATIVE_PY" case_studies/sql/eval_sql_seed334_test300.py
    ) >> "$ROOT/logs/baselines_14B/spider_itergen_20260612.log" 2>&1
    EXIT_itergen_spider14B=$?
fi
log "EXIT_itergen_spider14B=$EXIT_itergen_spider14B"

# B3 — Spider CARS (run_legacy_fixed_strategy)
log "=== 14B Spider cars $(date) ==="
SPIDER_CARS_OUT="$OUT_SPIDER_CARS/spider_seed334_test300__tb1__ms600.json"
if ! skip_if_exists "$SPIDER_CARS_OUT" "spider cars"; then
    "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
        --strategy cars --dataset spider \
        --eval-model "$MODEL" \
        --eval-sample-size 300 --eval-max-steps 600 --eval-step-token-budget 1 \
        --cars-search-steps 200 \
        --spider-split-file "$SSPLIT" --spider-split-name test \
        --output-json "$SPIDER_CARS_OUT" \
        >> "$ROOT/logs/baselines_14B/spider_cars_20260612.log" 2>&1
fi
EXIT_cars_spider14B=$?
log "EXIT_cars_spider14B=$EXIT_cars_spider14B"

# B4 — Spider GCD/SynCode (run_legacy_fixed_strategy, vllm)
log "=== 14B Spider gcd $(date) ==="
OUT_SPIDER_GCD=$ROOT/outputs/baselines/gcd/${M14_TAG}
mkdir -p "$OUT_SPIDER_GCD"
SPIDER_GCD_OUT="$OUT_SPIDER_GCD/spider_seed334_test300__tb1__ms600.json"
if ! skip_if_exists "$SPIDER_GCD_OUT" "spider gcd"; then
    "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
        --strategy gcd --dataset spider \
        --eval-model "$MODEL" --eval-backend vllm \
        --eval-sample-size 300 --eval-max-steps 600 --eval-step-token-budget 1 \
        --vllm-gpu-memory-utilization 0.85 --vllm-tensor-parallel-size 1 \
        --spider-split-file "$SSPLIT" --spider-split-name test \
        --output-json "$SPIDER_GCD_OUT" \
        >> "$ROOT/logs/baselines_14B/spider_gcd_20260612.log" 2>&1
fi
EXIT_gcd_spider14B=$?
log "EXIT_gcd_spider14B=$EXIT_gcd_spider14B"

# ── RESULTS ─────────────────────────────────────────────────────────────────────
log "=== Spider-14B RESULTS $(date) ==="
"$PY" - <<'PY' 2>/dev/null | tee -a "$MASTERLOG" || log "results print failed"
import json, glob, os

paths = [
    ("$ROOT/outputs/baselines/cars/Qwen_Qwen2.5_14B_Instruct/spider_seed334_test300__tb1__ms600.json", "cars"),
    ("$ROOT/outputs/baselines/gcd/Qwen_Qwen2.5_14B_Instruct/spider_seed334_test300__tb1__ms600.json", "gcd"),
]
for p, label in paths:
    try:
        d = json.load(open(p))
        print(f"  spider {label}: acc={d.get('accuracy',0):.3f} syn={d.get('syntax_rate',0):.3f} n={len(d.get('answers',[]))}")
    except Exception as e:
        print(f"  spider {label}: missing or error ({e})")

import os, glob
for mode in ("unconstrained", "itergen"):
    pattern = f"$ITERGEN_ROOT/results/sql_results/Qwen/Qwen2.5-14B-Instruct/{mode}*seed334*300.jsonl"
    files = glob.glob(pattern)
    if files:
        import subprocess
        n = subprocess.run(["wc", "-l", files[0]], capture_output=True, text=True).stdout.strip()
        print(f"  spider {mode} (native jsonl): {n}")
    else:
        print(f"  spider {mode} native: NOT FOUND (pattern={pattern})")
PY

log "=== DONE_14B_BASELINES $(date) ==="
log "EXIT SUMMARY:"
log "  gsm: unc=$EXIT_unconstrained_gsm14B gcd=$EXIT_gcd_gsm14B crane=$EXIT_crane_gsm14B itergen=$EXIT_itergen_gsm14B cars=$EXIT_cars_gsm14B"
log "  spider: unc=$EXIT_unconstrained_spider14B itergen=$EXIT_itergen_spider14B cars=$EXIT_cars_spider14B gcd=$EXIT_gcd_spider14B"
