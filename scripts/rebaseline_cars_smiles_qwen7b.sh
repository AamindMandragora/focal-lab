#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
OUT_DIR="outputs/baselines/cars/Qwen_Qwen2.5_Coder_7B_Instruct"
mkdir -p "$OUT_DIR"

for cls in acrylates chain_extenders isocyanates; do
  out="$OUT_DIR/smiles__class_${cls}__tb1__ms900__cs200.json"
  echo "=== CARS SMILES re-baseline: $cls -> $out ==="
  python -m synthesis.evaluate.run_legacy_fixed_strategy \
    --strategy cars \
    --dataset smiles \
    --eval-model "$MODEL" \
    --eval-backend huggingface \
    --device cuda \
    --eval-sample-size 100 \
    --eval-max-steps 900 \
    --eval-step-token-budget 1 \
    --cars-search-steps 200 \
    --smiles-classes "$cls" \
    --smiles-samples-per-class 100 \
    --output-json "$out"
  echo "=== Done $cls ==="
done

echo "All three CARS SMILES baselines complete."
