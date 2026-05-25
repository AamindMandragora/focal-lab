#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/csd-generation"
if [[ -f synthesis/.env ]]; then set -a; source synthesis/.env; set +a; fi
export CONDA_PREFIX="/apps/conda/advayth2/envs/advayth2"
export PATH="/apps/conda/advayth2/envs/advayth2/bin:${PATH:-}"
[[ -d "/apps/conda/advayth2/envs/advayth2/lib" ]] && export LD_LIBRARY_PATH="/apps/conda/advayth2/envs/advayth2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=2
export VAS_MAX_CUDA_DEVICES=1
export VAS_VLLM_GPU_MEMORY_UTILIZATION=0.65
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

ITER14_DIR="/home/aadivyar/csd-generation/outputs/generated/validation_3changes_metadecode_gsm_7b_opus47_iter15_20260520_224040_78a274/python/validation_3changes_metadecode_gsm_7b_opus47_iter15_20260521_001829_35d746/GeneratedCSD.py"
OUT_JSON="/home/aadivyar/csd-generation/outputs/generated/validation_3changes_metadecode_gsm_7b_opus47_iter15_20260520_224040_78a274/heldout_iter14_n100.json"
SPLIT="/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json"

echo "Started:  $(date -Is)"
echo "GPU:      CUDA_VISIBLE_DEVICES=2 (util=0.65)"
echo "Strategy: iter14 (acc=0.36, syn=0.915 on in-loop N=50)"
echo "Target:   N=100, CRANE eval split, ms=600, tb=400 (matches synth config)"
echo "----"
set +e
python -m synthesis.scripts.reevaluate_compiled_csd \
  "$ITER14_DIR" \
  --dataset gsm_symbolic \
  --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --eval-backend vllm \
  --device cuda \
  --sample-size 100 \
  --max-steps 600 \
  --step-token-budget 400 \
  --vllm-gpu-memory-utilization 0.65 \
  --gsm-split-file "$SPLIT" \
  --gsm-split-name eval \
  --output-json "$OUT_JSON"
EXIT=$?
echo "----"
echo "Exit: $EXIT  Finished: $(date -Is)"
echo "Output: $OUT_JSON"
