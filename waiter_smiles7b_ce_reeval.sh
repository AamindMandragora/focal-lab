#!/bin/bash
# chain_extenders-7B synthesis SUCCEEDED (train 1.0/1.0) but the running lane's stale
# in-memory glob missed the CSD. Re-eval it held-out (n=100) on GPU 2 as soon as the
# GSM-7B re-eval there finishes. Writing metadecode.json also makes the later backfill
# pass skip this class.
set -uo pipefail
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
CSD="outputs/generated/smiles_7B_chain_extenders_20260610/smiles_7B_chain_extenders_20260610_20260610_042525_e1e197/python/smiles_7B_chain_extenders_20260610_20260610_044142_57d7ed/GeneratedCSD.py"

until test -s outputs/controlled_comparison/gsm_7B/metadecode.json; do sleep 60; done
echo "GPU2_FREE $(date)"
test -s "$CSD" || { echo "CE_CSD_MISSING"; exit 1; }

python -m synthesis.scripts.reevaluate_compiled_csd "$CSD" \
  --dataset smiles --smiles-classes chain_extenders \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --sample-size 100 --max-steps 400 --step-token-budget 1 \
  --vllm-gpu-memory-utilization 0.45 \
  --output-json outputs/controlled_comparison/smiles_7B/chain_extenders/metadecode.json
echo "EXIT_REEVAL_CE7B=$?"
echo DONE_CE7B_REEVAL
