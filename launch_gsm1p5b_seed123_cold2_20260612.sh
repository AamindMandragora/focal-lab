#!/bin/bash
# Cold GSM-1.5B metaDecode synthesis on seed123 TRAIN split.
# Bar: CRANE 38.8% on non-Coder seed123 eval → --min-accuracy 0.41 (one step above).
# Cold run: no --initial-strategy-file. Full helper surface (--no-adaptive-helper-mask).
# Aligned metric + forbidden-token mask deployed on focal 2026-06-11.
# On acceptance, re-evaluates on the held-out EVAL side of the same split.
set -uo pipefail
cd /home/aadivyar/csd-generation

# Load .env credentials (AWS_BEARER_TOKEN_BEDROCK etc.)
set -a; source /home/aadivyar/csd-generation/.env; set +a

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

GSPLIT=environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
NAME=gsm1p5b_seed123_cold2_20260612
OUT=outputs/controlled_comparison/gsm_1p5B

echo "LAUNCH_SYNTH_GSM1P5B $(date)"

/apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters." \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name "$NAME" --output-dir "outputs/generated/$NAME" \
  --min-accuracy 0.41 --min-syntax-rate 0.90 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 \
  --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.40 --device auto --vllm-tensor-parallel-size 1 \
  --no-adaptive-helper-mask \
  --refinement-beam-size 2 \
  --gsm-split-file "$GSPLIT" --gsm-split-name train

SYNTH_EXIT=$?
echo "SYNTH_EXIT=$SYNTH_EXIT"

if [ "$SYNTH_EXIT" -eq 0 ]; then
  CSD=$(ls -t outputs/generated/${NAME}*/python/GeneratedCSD.py 2>/dev/null | head -1)
  if [ -n "$CSD" ]; then
    echo "REEVAL_START: $CSD $(date)"
    mkdir -p "$OUT"
    /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.scripts.reevaluate_compiled_csd "$CSD" \
      --dataset gsm_symbolic \
      --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
      --sample-size 49 --max-steps 900 --step-token-budget 1 \
      --vllm-gpu-memory-utilization 0.40 \
      --gsm-split-file "$GSPLIT" --gsm-split-name eval \
      --output-json "$OUT/metadecode.json"
    REEVAL_EXIT=$?
    echo "REEVAL_EXIT=$REEVAL_EXIT"
  else
    echo "NO_ACCEPTED_CSD_GSM1P5B"
    REEVAL_EXIT=1
  fi
else
  echo "SYNTH_FAILED_SKIP_REEVAL"
  REEVAL_EXIT=1
fi

echo "DONE_GSM1P5B_SEED123_COLD2"
