#!/bin/bash
# Finish the SMILES rolling-prompt A/B: wait for Run A (OFF) to end, run Run B (ON),
# rescore both. Holds temperature constant (0.7); only CSD_SMILES_ROLLING_PROMPT differs.
set -uo pipefail
cd /home/aadivyar/csd-generation
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=2
export CSD_CONSTRAINED_TEMPERATURE=0.7
CSD="outputs/generated/smiles_7B_chain_extenders_uvpilot_20260610/smiles_7B_chain_extenders_uvpilot_20260610_20260610_090648_2feb31/python/smiles_7B_chain_extenders_uvpilot_20260610_20260610_094544_38466a/GeneratedCSD.py"

echo "WAIT_A $(date)"
# Run A's process (launched separately) writes ce7b_OFF.json on success.
while ps -p 3740045 >/dev/null 2>&1; do sleep 3; done
echo "A_PROC_GONE $(date)"
if [ -s outputs/ab_rolling/ce7b_OFF.json ]; then
  echo "A_JSON_OK"
else
  echo "A_JSON_MISSING (Run A may have crashed; tail:)"
  tail -n 15 outputs/ab_rolling/run_A_off.log
fi

echo "START_B $(date)"
export CSD_SMILES_ROLLING_PROMPT=1
python -m synthesis.scripts.reevaluate_compiled_csd "$CSD" \
  --dataset smiles --smiles-classes chain_extenders \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --sample-size 100 --max-steps 400 --step-token-budget 1 \
  --vllm-gpu-memory-utilization 0.45 \
  --output-json outputs/ab_rolling/ce7b_ON.json > outputs/ab_rolling/run_B_on.log 2>&1
echo "EXIT_B=$?"

echo "=== ACCURACY FIELDS (unique-valid rate straight from each JSON) ==="
python - <<'PY'
import json
for tag, p in [("OFF", "outputs/ab_rolling/ce7b_OFF.json"), ("ON", "outputs/ab_rolling/ce7b_ON.json")]:
    try:
        d = json.load(open(p))
        ans = d.get("answers", [])
        print(f"{tag}: accuracy(uniq-valid)={d.get('accuracy')} syntax={d.get('syntax_rate')} N={len(ans)}")
    except Exception as e:
        print(f"{tag}: ERROR {e}")
PY

echo "=== RESCORE (unique-valid / validity / diversity) ==="
python rescore_smiles_unique_valid.py outputs/ab_rolling chain_extenders 2>&1 || echo "RESCORE_FAILED"
echo "DONE_AB $(date)"
