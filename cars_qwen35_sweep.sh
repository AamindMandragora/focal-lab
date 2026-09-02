#!/bin/bash
# CARS baseline sweep for Qwen3.5 (2B, 4B, 9B) x (acrylates, chain_extenders, isocyanates)
# GPU 3 only. Sequential. N=50 per cell (--smiles-samples-per-class 50 --eval-sample-size 50).
# Usage: bash cars_qwen35_sweep.sh
set -uo pipefail
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}

OUTBASE="outputs/controlled_comparison/smiles_qwen35"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=== Qwen3.5 CARS sweep start $STAMP ==="

for MODEL in "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B" "Qwen/Qwen3.5-9B"; do
  # Tag: 2B, 4B, 9B
  TAG="${MODEL##*/}"   # Qwen3.5-2B etc
  SIZE="${TAG##*-}"    # 2B, 4B, 9B

  for CLASS in acrylates chain_extenders isocyanates; do
    OUTDIR="$OUTBASE/${SIZE}/${CLASS}"
    mkdir -p "$OUTDIR"
    OUTJSON="$OUTDIR/cars.json"

    if [ -s "$OUTJSON" ]; then
      echo "SKIP existing $OUTJSON"
      continue
    fi

    echo "=== cars smiles/$CLASS $MODEL START $(date) ==="
    /apps/conda/aadivyar/envs/csd/bin/python -m synthesis.evaluate.run_legacy_fixed_strategy \
      --strategy cars \
      --dataset smiles \
      --eval-model "$MODEL" \
      --eval-backend vllm \
      --smiles-classes "$CLASS" \
      --smiles-samples-per-class 50 \
      --eval-sample-size 50 \
      --eval-max-steps 400 \
      --eval-step-token-budget 1 \
      --cars-search-steps 200 \
      --vllm-gpu-memory-utilization 0.85 \
      --device auto \
      --output-json "$OUTJSON"
    EC=$?
    echo "EXIT_cars_${SIZE}_${CLASS}=$EC"
    echo ""
  done
done

echo "=== All cells done $(date) ==="
echo "=== Stored membership accuracies (NOT the real UV metric) ==="
/apps/conda/aadivyar/envs/csd/bin/python - "$OUTBASE" <<'PY'
import json, sys, glob
base = sys.argv[1]
for p in sorted(glob.glob(f"{base}/*/*.json")):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    parts = p.replace(base+"/", "").split("/")
    size, cls = parts[0], parts[1].replace(".json","")
    print(f"  {size:6} {cls:20} acc={d.get('accuracy', '?'):.3f}  syn={d.get('syntax_rate', '?'):.3f}  N={len(d.get('answers',[]))}")
PY

echo "SWEEP_DONE"
