#!/bin/bash
# Re-run the gsm_7B baseline cell (non-Coder Instruct, seed123 eval) after the disk-full crash.
# GSM only — Spider is out of the active goal. Runs 4 CRANE-repo baselines then CARS (vocab-fix applied).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
set -uo pipefail
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
GSPLIT="$SPLITS_DIR/gsm_symbolic_crane_proportional_49x49_seed123.json"
M7=Qwen/Qwen2.5-7B-Instruct
OUT=outputs/controlled_comparison/gsm_7B

for S in unconstrained itergen crane gcd; do
  echo "=== $S gsm_symbolic $M7 START $(date) ==="
  $PY -m synthesis.evaluate.run_legacy_fixed_strategy \
    --strategy "$S" --dataset gsm_symbolic --eval-model "$M7" \
    --eval-backend huggingface \
    --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
    --gsm-split-file "$GSPLIT" --gsm-split-name eval \
    --output-json "$OUT/$S.json"
  echo "EXIT_${S}_gsm7B=$?"
done

echo "=== cars gsm_symbolic $M7 START $(date) ==="
$PY -m synthesis.evaluate.run_legacy_fixed_strategy \
  --strategy cars --dataset gsm_symbolic --eval-model "$M7" \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --cars-search-steps 200 \
  --gsm-split-file "$GSPLIT" --gsm-split-name eval \
  --output-json "$OUT/cars.json"
echo "EXIT_cars_gsm7B=$?"

echo "=================== RESULTS $(date) ==================="
$PY - <<'PY'
import json, glob
for p in sorted(glob.glob("outputs/controlled_comparison/gsm_7B/*.json")):
    d = json.load(open(p)); ans = d.get("answers", [])
    empty = sum(1 for a in ans if not str(a.get("generated_answer", "")).strip())
    print(p.split("gsm_7B/")[1], "acc=", round(d.get("accuracy", -1), 3),
          "syn=", round(d.get("syntax_rate", -1), 3), "n=", len(ans), "empty=", empty)
PY
echo DONE_GSM7B_BASELINES
