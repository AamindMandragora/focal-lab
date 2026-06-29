#!/bin/bash
# SMILES per-class lane: baselines(n=100) -> bar -> metaDecode synthesis(train-50) -> CSD held-out(n=100).
# Replaces the deleted scripts/smiles_generalization_workflow.py orchestrator using master tooling.
# Usage: smiles_lane.sh <eval-model> <tag> <gpu> <vllm-util>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
set -uo pipefail
MODEL="$1"; TAG="$2"; GPU="$3"; UTIL="$4"
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
BASE="outputs/controlled_comparison/smiles_${TAG}"
STAMP=20260610

for CLASS in acrylates chain_extenders isocyanates; do
  OUT="$BASE/$CLASS"
  mkdir -p "$OUT"

  for S in cars itergen gcd unconstrained; do
    if [ -s "$OUT/$S.json" ]; then echo "SKIP existing $OUT/$S.json"; continue; fi
    echo "=== $S smiles/$CLASS $MODEL START $(date) ==="
    $PY -m synthesis.evaluate.run_legacy_fixed_strategy \
      --strategy "$S" --dataset smiles --eval-model "$MODEL" \
      --smiles-classes "$CLASS" --eval-sample-size 100 \
      --eval-max-steps 400 --eval-step-token-budget 1 \
      --cars-search-steps 200 \
      --vllm-gpu-memory-utilization "$UTIL" \
      --output-json "$OUT/$S.json"
    echo "EXIT_${S}_${CLASS}=$?"
  done

  # A finished class has a metadecode.json from the held-out re-eval; on relaunch
  # only backfill classes that never got that far (baselines above still skip-if-exists).
  if [ -s "$OUT/metadecode.json" ]; then
    echo "SKIP synthesis for $CLASS (metadecode.json exists)"
    continue
  fi

  # Bar: beat best baseline by >=1 example; BOTH capped at 0.95 (never 1.0 — a 1.0
  # threshold makes acceptance impossible and triggers threshold-impossible early-stop,
  # e.g. CARS acrylates-7B sits at acc 1.0). Win is still decided at held-out re-eval.
  read MINACC MINSYN <<<"$($PY - "$OUT" <<'PY'
import json, sys, glob
out = sys.argv[1]
best_acc, best_syn = 0.0, 0.0
for p in glob.glob(f"{out}/*.json"):
    if p.endswith("metadecode.json"): continue
    try: d = json.load(open(p))
    except Exception: continue
    best_acc = max(best_acc, float(d.get("accuracy") or 0.0))
    best_syn = max(best_syn, float(d.get("syntax_rate") or 0.0))
nxt = lambda r, n: min(1.0, (int(round(r * n)) + 1) / n)
print(f"{min(0.95, nxt(best_acc, 100)):.4f} {min(0.95, nxt(best_syn, 100)):.4f}")
PY
)"
  echo "BAR_${CLASS}: min-accuracy=$MINACC min-syntax=$MINSYN"

  NAME="smiles_${TAG}_${CLASS}_${STAMP}"

  # Accepted-strategy pickup: rejected attempts also leave GeneratedCSD.py files, so a
  # bare glob is wrong (and misses the nested run dir anyway). The reliable pointer is
  # latest_run.txt -> results/success_report.json -> compiled_dir. If a previous pass
  # already accepted a strategy, reuse it instead of burning another synthesis run.
  pick_accepted_csd() {
    local rundir cdir
    rundir=$(cat "outputs/generated/$NAME/latest_run.txt" 2>/dev/null)
    [ -n "$rundir" ] && [ -s "$rundir/results/success_report.json" ] || return 0
    cdir=$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('compiled_dir',''))" "$rundir/results/success_report.json" 2>/dev/null)
    [ -n "$cdir" ] && [ -s "$cdir/GeneratedCSD.py" ] && echo "$cdir/GeneratedCSD.py"
  }

  CSD=$(pick_accepted_csd)
  if [ -n "$CSD" ]; then
    echo "REUSE accepted CSD for $CLASS: $CSD"
  else
  echo "=== SYNTHESIS smiles/$CLASS $MODEL START $(date) ==="
  $PY -m synthesis.run_synthesis \
    --task "Generate one new, valid, non-exemplar SMILES molecule for the ${CLASS} class. The answer contract is a single SMILES string and nothing else. Use the hidden parser-guided constrained chunk for that SMILES token sequence and avoid copying prompt exemplars." \
    --dataset smiles --smiles-classes "$CLASS" --smiles-samples-per-class 50 \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --anthropic-thinking enabled --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --eval-model "$MODEL" --eval-backend vllm \
    --max-iterations 15 \
    --output-name "$NAME" --output-dir "outputs/generated/$NAME" \
    --min-accuracy "$MINACC" --min-syntax-rate "$MINSYN" \
    --eval-max-steps 400 --eval-step-token-budget 1 \
    --eval-min-examples-before-threshold-stop 50 \
    --max-tokens 32768 \
    --restart-after-stuck-iters 0 \
    --vllm-gpu-memory-utilization "$UTIL" --device auto --vllm-tensor-parallel-size 1 \
    --adaptive-helper-mask --helper-selection-policy bandit \
    --refinement-beam-size 2
  echo "EXIT_SYNTH_${CLASS}=$?"
  CSD=$(pick_accepted_csd)
  fi

  if [ -n "$CSD" ]; then
    echo "=== CSD HELD-OUT smiles/$CLASS $MODEL START $(date) ==="
    $PY -m synthesis.scripts.reevaluate_compiled_csd "$CSD" \
      --dataset smiles --smiles-classes "$CLASS" \
      --eval-model "$MODEL" --eval-backend vllm \
      --sample-size 100 --max-steps 400 --step-token-budget 1 \
      --vllm-gpu-memory-utilization "$UTIL" \
      --output-json "$OUT/metadecode.json"
    echo "EXIT_REEVAL_${CLASS}=$?"
  else
    echo "NO_ACCEPTED_CSD_${CLASS} (synthesis did not produce a compiled strategy)"
  fi
done

echo "=================== RESULTS smiles_${TAG} $(date) ==================="
$PY - "$BASE" <<'PY'
import json, sys, glob
for p in sorted(glob.glob(f"{sys.argv[1]}/*/*.json")):
    try: d = json.load(open(p))
    except Exception: continue
    print(p.split("controlled_comparison/")[1],
          "acc=", round(d.get("accuracy", -1), 3),
          "syn=", round(d.get("syntax_rate", -1), 3),
          "n=", len(d.get("answers", [])))
PY
echo "DONE_SMILES_LANE_${TAG}"
