#!/bin/bash
# Generic warmstart synthesis retry for a single SMILES class that failed the first pass
# on the single-atom trap. Continues from the author's OWN best near-miss strategy with a
# larger iteration budget (25), then re-evals held-out (n=100). This is the exact pattern
# that turned acrylates-1.5B from 15/15-fail into a 1.0/1.0 win.
# Usage: warmstart_retry.sh <eval-model> <tag> <class> <gpu> <util> <min-acc> <min-syn> <warmstart-file>
set -uo pipefail
MODEL="$1"; TAG="$2"; CLASS="$3"; GPU="$4"; UTIL="$5"; MINACC="$6"; MINSYN="$7"; WARM="$8"
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
NAME="smiles_${TAG}_${CLASS}_retry_20260610"
OUT="outputs/controlled_comparison/smiles_${TAG}/${CLASS}"
mkdir -p "$OUT"

if [ -s "$OUT/metadecode.json" ]; then echo "SKIP ${TAG}/${CLASS} retry (metadecode.json exists)"; echo "DONE_RETRY_${TAG}_${CLASS}"; exit 0; fi
test -s "$WARM" || { echo "WARMSTART_FILE_MISSING $WARM"; exit 1; }

echo "=== WARMSTART RETRY smiles/${CLASS} ${MODEL} START $(date) (warm=$WARM bar=$MINACC/$MINSYN) ==="
python -m synthesis.run_synthesis \
  --task "Generate one new, valid, non-exemplar SMILES molecule for the ${CLASS} class. The answer contract is a single SMILES string and nothing else. Use the hidden parser-guided constrained chunk for that SMILES token sequence and avoid copying prompt exemplars." \
  --dataset smiles --smiles-classes "$CLASS" --smiles-samples-per-class 50 \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --anthropic-thinking-display summarized \
  --eval-model "$MODEL" --eval-backend vllm \
  --max-iterations 25 \
  --initial-strategy-file "$WARM" \
  --output-name "$NAME" --output-dir "outputs/generated/$NAME" \
  --min-accuracy "$MINACC" --min-syntax-rate "$MINSYN" \
  --eval-max-steps 400 --eval-step-token-budget 1 \
  --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization "$UTIL" --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --refinement-beam-size 2
echo "EXIT_SYNTH_${TAG}_${CLASS}_retry=$?"

RUNDIR=$(cat "outputs/generated/$NAME/latest_run.txt" 2>/dev/null)
CSD=""
if [ -n "$RUNDIR" ] && [ -s "$RUNDIR/results/success_report.json" ]; then
  CDIR=$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('compiled_dir',''))" "$RUNDIR/results/success_report.json" 2>/dev/null)
  [ -n "$CDIR" ] && [ -s "$CDIR/GeneratedCSD.py" ] && CSD="$CDIR/GeneratedCSD.py"
fi
if [ -n "$CSD" ]; then
  echo "ACCEPTED_CSD_${TAG}_${CLASS}=$CSD"
  python -m synthesis.scripts.reevaluate_compiled_csd "$CSD" \
    --dataset smiles --smiles-classes "$CLASS" \
    --eval-model "$MODEL" --eval-backend vllm \
    --sample-size 100 --max-steps 400 --step-token-budget 1 \
    --vllm-gpu-memory-utilization "$UTIL" \
    --output-json "$OUT/metadecode.json"
  echo "EXIT_REEVAL_${TAG}_${CLASS}_retry=$?"
else
  echo "NO_ACCEPTED_CSD_${TAG}_${CLASS}_retry"
fi
echo "DONE_RETRY_${TAG}_${CLASS}"
