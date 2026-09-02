#!/bin/bash
# Unconstrained Spider baselines on the exact seed334 test-300 split (2026-06-11).
# Runs AFTER the prompt-type bug fix in run_legacy_fixed_strategy.py
# (run_unconstrained_spider_adapter passed a list[dict] chat prompt to vLLM; now uses
# the flat few-shot "evaluator_default" string — TDD red/green + 3-example smoke 0.667/1.0).
# Sequential: 1.5B then 7B, both on GPU 1 (CARS-7B finished; ~22GB free at util 0.20).
set -u
cd /home/aadivyar/csd-generation
SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

for MODEL in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  # 1.5B fits in 0.20 (8GB); 7B weights alone are ~15GB -> needs ~0.45 (18GB; GPU 1 has ~22GB free)
  UTIL=0.20; [ "$MODEL" = "Qwen/Qwen2.5-7B-Instruct" ] && UTIL=0.45
  # match the existing baselines dir naming convention (Qwen_Qwen2.5_7B_Instruct)
  OUTDIR="outputs/baselines/unconstrained/$(echo "$MODEL" | sed 's#/#_#g' | sed 's/-/_/g')"
  mkdir -p "$OUTDIR"
  echo "=== UNCONSTRAINED SPIDER $MODEL $(date) ==="
  CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} \
  /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.evaluate.run_legacy_fixed_strategy \
    --strategy unconstrained --dataset spider \
    --eval-model "$MODEL" --eval-backend vllm --device auto \
    --eval-sample-size 300 --eval-max-steps 600 --eval-step-token-budget 1 \
    --vllm-gpu-memory-utilization "$UTIL" --vllm-tensor-parallel-size 1 \
    --spider-split-file "$SPLIT" --spider-split-name test \
    --output-json "$OUTDIR/spider_seed334_test300_unconstrained_fixedprompt.json"
  echo "EXIT_UNC_${MODEL}=$?"
done
echo "DONE_UNCONSTRAINED_SPIDER_BOTH $(date)"
