#!/bin/bash
# acrylates-1.5B synthesis failed 15/15: the 1.5B emits the single atom "C" inside the
# span and every prefix-complete molecule ends the span, so acc=0.0 at syn=1.0. The author
# had correctly diagnosed this by attempt 13-15 (was building a min-length mechanism but
# hit the AppendConstrainedToken !IsCompletePrefix precondition). Retry = continue from
# its own attempt-15 strategy with a bigger iteration budget, after the 1.5B lane ends.
set -uo pipefail
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
LOG=outputs/smiles_lane_1p5B.log
NAME=smiles_1p5B_acrylates_retry_20260610
OUT=outputs/controlled_comparison/smiles_1p5B/acrylates

until grep -q DONE_SMILES_LANE_1P5B "$LOG" 2>/dev/null; do sleep 120; done
echo "LANE_1P5B_DONE $(date)"

# Skip if a backfill/other pass already produced the result.
if [ -s "$OUT/metadecode.json" ]; then echo "SKIP retry (metadecode.json exists)"; exit 0; fi

python -m synthesis.run_synthesis \
  --task "Generate one new, valid, non-exemplar SMILES molecule for the acrylates class. The answer contract is a single SMILES string and nothing else. Use the hidden parser-guided constrained chunk for that SMILES token sequence and avoid copying prompt exemplars." \
  --dataset smiles --smiles-classes acrylates --smiles-samples-per-class 50 \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 25 \
  --initial-strategy-file warmstart_smiles1p5b_acrylates_att15.dfy \
  --output-name "$NAME" --output-dir "outputs/generated/$NAME" \
  --min-accuracy 0.9300 --min-syntax-rate 0.9500 \
  --eval-max-steps 400 --eval-step-token-budget 1 \
  --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.30 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --refinement-beam-size 2
echo "EXIT_SYNTH_acrylates_retry=$?"

RUNDIR=$(cat "outputs/generated/$NAME/latest_run.txt" 2>/dev/null)
CSD=""
if [ -n "$RUNDIR" ] && [ -s "$RUNDIR/results/success_report.json" ]; then
  CDIR=$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('compiled_dir',''))" "$RUNDIR/results/success_report.json" 2>/dev/null)
  [ -n "$CDIR" ] && [ -s "$CDIR/GeneratedCSD.py" ] && CSD="$CDIR/GeneratedCSD.py"
fi
if [ -n "$CSD" ]; then
  python -m synthesis.scripts.reevaluate_compiled_csd "$CSD" \
    --dataset smiles --smiles-classes acrylates \
    --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
    --sample-size 100 --max-steps 400 --step-token-budget 1 \
    --vllm-gpu-memory-utilization 0.30 \
    --output-json "$OUT/metadecode.json"
  echo "EXIT_REEVAL_acrylates_retry=$?"
else
  echo "NO_ACCEPTED_CSD_acrylates_retry"
fi
echo DONE_ACRYLATES_1P5B_RETRY
