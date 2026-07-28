#!/bin/bash
# BASELINES-ONLY variant of smiles_lane.sh (2026-07-03): runs the per-class baseline
# evals (cars/itergen/gcd/unconstrained, N=100, skip-if-exists) and the results print,
# and NOTHING else — the synthesis + held-out blocks of the original are removed so
# this can never start a billed author run.
# Usage: smiles_lane_baselines_only.sh <eval-model> <tag> <gpu> <vllm-util>
set -uo pipefail
MODEL="$1"; TAG="$2"; GPU="$3"; UTIL="$4"
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
# Qwen3.5 KV-cache bug guard: any HF-backend path must not reuse the cache.
export CSD_HF_KV_CACHE=0
# Qwen3.5 model code needs the csd env python (transformers 5.5.4), not bare `python`.
PY=/apps/conda/aadivyar/envs/csd/bin/python
BASE="outputs/controlled_comparison/smiles_${TAG}"

for CLASS in acrylates chain_extenders isocyanates; do
  OUT="$BASE/$CLASS"
  mkdir -p "$OUT"
  for S in cars itergen gcd unconstrained; do
    if [ -s "$OUT/$S.json" ]; then echo "SKIP existing $OUT/$S.json"; continue; fi
    echo "=== $S smiles/$CLASS $MODEL START $(date) ==="
    "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
      --strategy "$S" --dataset smiles --eval-model "$MODEL" \
      --smiles-classes "$CLASS" --eval-sample-size 100 \
      --eval-max-steps 400 --eval-step-token-budget 1 \
      --cars-search-steps 200 \
      --vllm-gpu-memory-utilization "$UTIL" \
      --output-json "$OUT/$S.json"
    echo "EXIT_${S}_${CLASS}=$?"
  done
done

echo "=================== RESULTS smiles_${TAG} $(date) ==================="
"$PY" - "$BASE" <<'PY'
import json, sys, glob
for p in sorted(glob.glob(f"{sys.argv[1]}/*/*.json")):
    try: d = json.load(open(p))
    except Exception: continue
    print(p.split("controlled_comparison/")[1],
          "acc=", round(d.get("accuracy", -1), 3),
          "syn=", round(d.get("syntax_rate", -1), 3),
          "n=", len(d.get("answers", [])))
PY
echo "DONE_SMILES_BASELINES_${TAG}"
