#!/usr/bin/env bash
# AUTONOMOUS WATCHER (user asleep 2026-06-24): wait for the live 1.5B 300-train COLD run to finish,
# harvest its BEST/winning strategy, then auto-launch a $0 held-out TEST-300 re-eval on it.
# Mirrors the 7B att31 pipeline exactly. Runs via nohup; independent of the Claude session.
#
# User instruction: "ensure that as soon as the 1.5b win registers you relaunch a re-eval on that too".
# Interpretation: fire when the run ENDS. If it crossed its 0.60 accept bar -> success_report.json
# (a real WIN); else -> failure_report.json and we take the best attempt (same as the 7B att31 case,
# which the user wanted re-eval'd despite being a failure_report best). Provenance is logged either way.
#
# $0 Bedrock: re-eval uses --max-iterations 1 + --initial-strategy-file => Sonnet author never called.
# Fair: 1.5B iterated only on train_indices; test_indices disjoint (overlap 0, verified). Eval recipe
# (util 0.30, eval-max-steps 200, token-0, mask ON + bandit, no-require-delimiters) matches the train run.
set -u
cd ~/csd-generation
set -a; source ~/csd-generation/.env 2>/dev/null; set +a
export SPIDER_DB_DIR=~/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
RUN_NAME=spider1p5b_300train_cold_iter40_20260624_080442
WLOG=logs/auto_reeval_1p5b_watcher.log
BODY=saved-results/spider-goal-strategies/best_1p5b_300train_body.dfy
SPLIT=environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json
TASK='Generate a single valid SQL query as exactly SQL: YOUR QUERY, using only the provided schema context.'
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$WLOG"; }

log "watcher START; waiting for run '$RUN_NAME' to exit (poll 120s)"
while pgrep -f "$RUN_NAME" >/dev/null 2>&1; do sleep 120; done
log "1.5B run EXITED"
sleep 30   # let vLLM release GPU memory

OUTDIR=$(ls -dt outputs/generated/${RUN_NAME}_* 2>/dev/null | head -1)
log "1.5B output dir: ${OUTDIR:-NONE}"
if [ -z "${OUTDIR:-}" ]; then log "FATAL: no output dir found; abort"; exit 1; fi

# Harvest best/winning strategy body
python3 - "$OUTDIR" "$BODY" <<'PY' 2>&1 | tee -a "$WLOG"
import sys, json, os, ijson
outdir, body = sys.argv[1], sys.argv[2]
succ = os.path.join(outdir,'results','success_report.json')
fail = os.path.join(outdir,'results','failure_report.json')
strat=prov=acc=syn=num=None
if os.path.exists(succ):
    d=json.load(open(succ)); strat=d.get('strategy_code'); prov='success_report (WIN)'
    er=d.get('evaluation_result',{}) or {}; acc=er.get('accuracy'); syn=er.get('syntax_rate')
elif os.path.exists(fail):
    best=None
    with open(fail,'rb') as f:
        for att in ijson.items(f,'attempts.item'):
            ev=att.get('evaluation') or {}
            a=ev.get('accuracy')
            if a is not None and (best is None or a>best[0]):
                best=(a, att.get('strategy_code'), att.get('attempt_number'), ev.get('syntax_rate'))
    if best: acc,strat,num,syn=best; prov=f'failure_report best (attempt {num}, NO WIN)'
if strat and strat.lstrip().startswith('// CSD_RATIONALE_BEGIN'):
    open(body,'w').write(strat)
    print(f"HARVEST_OK provenance='{prov}' train_acc={acc} train_syn={syn} -> {body} ({len(strat)} chars)")
else:
    print(f"HARVEST_FAIL provenance='{prov}' strat_present={bool(strat)} (not CSD body or missing)")
PY

if [ ! -s "$BODY" ] || ! head -1 "$BODY" | grep -q "CSD_RATIONALE_BEGIN"; then
  log "FATAL: harvested body invalid/empty; abort (no re-eval launched)"; exit 1
fi
log "harvested body md5 $(md5sum "$BODY" | cut -d' ' -f1)"

# Pick a free GPU (>=20GB free); fall back to GPU 2 (the one the 1.5B just freed)
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk '$2>20000{print $1; exit}')
GPU=${GPU:-2}
log "selected GPU=$GPU"

STAMP=$(date +%Y%m%d_%H%M%S)
NAME=spider1p5b_bestrun_reeval_300TEST_seed334_${STAMP}
LOG=logs/${NAME}.log
log "launching $NAME on GPU $GPU -> $LOG (PURE RE-EVAL, max-iter 1, \$0 Bedrock)"
CUDA_VISIBLE_DEVICES=$GPU python -m synthesis.run_synthesis \
  --task "$TASK" --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --initial-strategy-file "$BODY" \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 1 \
  --min-accuracy 0.0 --min-syntax-rate 0.0 --no-require-delimiters \
  --eval-sample-size 300 --eval-min-examples-before-threshold-stop 300 \
  --eval-max-steps 200 --eval-step-token-budget 1 --eval-max-seconds-per-example 450 \
  --restart-after-stuck-iters 0 \
  --vllm-max-model-len 16384 \
  --vllm-gpu-memory-utilization 0.30 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --spider-split-file "$SPLIT" --spider-split-name test \
  --output-name "$NAME" \
  > "$LOG" 2>&1
log "[done] $NAME exit=$? -> $LOG"
