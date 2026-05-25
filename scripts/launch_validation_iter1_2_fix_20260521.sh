#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/csd-generation"
if [[ -f synthesis/.env ]]; then set -a; source synthesis/.env; set +a; fi
export CONDA_PREFIX="/apps/conda/advayth2/envs/advayth2"
export PATH="/apps/conda/advayth2/envs/advayth2/bin:${PATH:-}"
[[ -d "/apps/conda/advayth2/envs/advayth2/lib" ]] && export LD_LIBRARY_PATH="/apps/conda/advayth2/envs/advayth2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=2
export VAS_VLLM_GPU_MEMORY_UTILIZATION=0.65
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
echo "Started:  $(date -Is)"
echo "GPU:      CUDA_VISIBLE_DEVICES=2 (util=0.65)"
echo "Purpose:  validation of simplified eval refinement prompts (iter-by-iter trajectory)"
echo "Config:   1.5B eval, opus4.7 generator, n=25, 10 iters, tb=1, ms=600 (mirrors prior failing run)"
echo "----"
set +e
python -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters." \
  --dataset gsm_symbolic \
  --generation-model claude-opus-4-7 \
  --generation-backend anthropic \
  --eval-model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eval-backend vllm \
  --max-iterations 10 \
  --output-name validation_iter12_fix_metadecode_gsm_1p5b_opus47_iter10 \
  --min-accuracy 0.31 \
  --min-syntax-rate 0.90 \
  --eval-sample-size 25 \
  --eval-max-steps 600 \
  --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 \
  --eval-min-examples-before-threshold-stop 15 \
  --vllm-gpu-memory-utilization 0.65 \
  --device auto \
  --output-dir outputs/generated
EXIT=$?
echo "----"
echo "Exit: $EXIT  Finished: $(date -Is)"
