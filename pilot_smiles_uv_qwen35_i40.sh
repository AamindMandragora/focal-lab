#!/bin/bash
# Qwen3.5 SMILES synthesis against the unique-valid-rate (UV) objective.
# Adapted from pilot_smiles_uv.sh for the Qwen3.5 small series:
#   - uses the csd conda env python + lib (Qwen3.5 needs transformers 5.5.4 / vllm 0.19.1)
#   - sources .env for Bedrock author creds (account 887730490125)
# Everything else identical to the proven pilot: UV metric (eval_logic.override_accuracy),
# temp 0.7 (sample within molecule span), COLD (no warmstart), Sonnet-4.6 author thinking-high,
# adaptive helper mask + bandit, auto held-out re-eval of the accepted CSD on N=100.
# Usage: pilot_smiles_uv_qwen35.sh <eval-model> <tag> <class> <gpu> <util> <min-acc> <min-syn>
set -uo pipefail
MODEL="$1"; TAG="$2"; CLASS="$3"; GPU="$4"; UTIL="$5"; MINACC="$6"; MINSYN="$7"
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=/apps/conda/aadivyar/envs/csd/lib:${LD_LIBRARY_PATH:-}
export CSD_CONSTRAINED_TEMPERATURE=0.7      # sample within the molecule span
PY=/apps/conda/aadivyar/envs/csd/bin/python
set -a; source /home/aadivyar/csd-generation/.env; set +a   # Bedrock creds (887730490125)
NAME="smiles_${TAG}_${CLASS}_uv_qwen35_0627"
OUT="outputs/controlled_comparison/smiles_${TAG}/${CLASS}"
mkdir -p "$OUT"

echo "=== UV-Qwen3.5 smiles/${CLASS} ${MODEL} START $(date) (temp=0.7 bar=${MINACC} uniq-valid / ${MINSYN} validity) ==="
"$PY" -m synthesis.run_synthesis \
  --task "Generate one new, valid, non-exemplar SMILES molecule for the ${CLASS} class. The answer contract is a single SMILES string and nothing else. Use the hidden parser-guided constrained chunk for that SMILES token sequence and avoid copying prompt exemplars." \
  --dataset smiles --smiles-classes "$CLASS" --smiles-samples-per-class 50 \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --anthropic-thinking-display summarized \
  --eval-model "$MODEL" --eval-backend vllm \
  --max-iterations 40 \
  --output-name "$NAME" --output-dir "outputs/generated/$NAME" \
  --min-accuracy "$MINACC" --min-syntax-rate "$MINSYN" \
  --eval-max-steps 400 --eval-step-token-budget 1 \
  --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization "$UTIL" --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --refinement-beam-size 2
echo "EXIT_SYNTH_${TAG}_${CLASS}_uv=$?"

RUNDIR=$(cat "outputs/generated/$NAME/latest_run.txt" 2>/dev/null)
CSD=""
if [ -n "$RUNDIR" ] && [ -s "$RUNDIR/results/success_report.json" ]; then
  CDIR=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('compiled_dir',''))" "$RUNDIR/results/success_report.json" 2>/dev/null)
  [ -n "$CDIR" ] && [ -s "$CDIR/GeneratedCSD.py" ] && CSD="$CDIR/GeneratedCSD.py"
fi
if [ -n "$CSD" ]; then
  echo "ACCEPTED_CSD_${TAG}_${CLASS}_uv=$CSD"
  "$PY" -m synthesis.scripts.reevaluate_compiled_csd "$CSD" \
    --dataset smiles --smiles-classes "$CLASS" \
    --eval-model "$MODEL" --eval-backend vllm \
    --sample-size 100 --max-steps 400 --step-token-budget 1 \
    --vllm-gpu-memory-utilization "$UTIL" \
    --output-json "$OUT/metadecode_uv.json"
  echo "EXIT_REEVAL_${TAG}_${CLASS}_uv=$?"
else
  echo "NO_ACCEPTED_CSD_${TAG}_${CLASS}_uv"
fi
echo "DONE_UV_${TAG}_${CLASS}_SENTINEL"
