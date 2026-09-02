#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/csd-generation"
if [[ -f synthesis/.env ]]; then set -a; source synthesis/.env; set +a; fi
export CONDA_PREFIX="/apps/conda/advayth2/envs/advayth2"
export PATH="/apps/conda/advayth2/envs/advayth2/bin:${PATH:-}"
[[ -d "/apps/conda/advayth2/envs/advayth2/lib" ]] && export LD_LIBRARY_PATH="/apps/conda/advayth2/envs/advayth2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=3
export VAS_MAX_CUDA_DEVICES=1
export VAS_VLLM_GPU_MEMORY_UTILIZATION=0.65
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
echo "Started:  $(date -Is)"
echo "GPU:      CUDA_VISIBLE_DEVICES=3 (util=0.65)"
echo "Bundle:   fixed-seed + Pareto-anchor + total-accuracy objective + RESTART mechanism"
echo "Restart:  after-stuck-iters=1, cooldown-iters=0 (very aggressive exploration)"
echo "Phase:    machinery removed (single objective: total accuracy)"
echo "----"
set +e
python -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters." \
  --dataset gsm_symbolic \
  --generation-model claude-opus-4-7 \
  --generation-backend anthropic \
  --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --eval-backend vllm \
  --max-iterations 15 \
  --output-name validation_restart_metadecode_gsm_7b_opus47_iter15 \
  --min-accuracy 0.39 \
  --min-syntax-rate 0.94 \
  --restart-after-stuck-iters 1 \
  --restart-cooldown-iters 0 \
  --eval-sample-size 50 \
  --eval-max-steps 600 \
  --eval-step-token-budget 400 \
  --eval-max-seconds-per-example 90 \
  --eval-min-examples-before-threshold-stop 25 \
  --vllm-gpu-memory-utilization 0.65 \
  --device auto \
  --output-dir outputs/generated
EXIT=$?
echo "----"
echo "Exit: $EXIT  Finished: $(date -Is)"
