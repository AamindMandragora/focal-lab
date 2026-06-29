#!/bin/bash
# Resume GSM-1.5B synthesis after the 2026-06-10 grading-hang kill.
# Same command as the seed123 fresh run, plus:
#   - warm start from the attempt-3 anchor (acc 30.6% / syn 55.1%), reconstructed
#     from its compiled Python (Dafny verify gates it at launch)
#   - evaluator.py now runs scoring under the per-example timer, so a wedged
#     grading step times out at 120s instead of hanging the run
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
NAME="gsm1p5b_seed123_warm3_20260610"

echo "=== GSM1P5B WARM3 RESUME START $(date) (warm-start attempt3 anchor, grading under timer) ==="
$PY -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters." \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name "$NAME" --output-dir "outputs/generated/$NAME" \
  --min-accuracy 0.41 --min-syntax-rate 0.90 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 120 \
  --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.30 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --refinement-beam-size 2 \
  --gsm-split-file "$SPLITS_DIR/gsm_symbolic_crane_proportional_49x49_seed123.json" \
  --gsm-split-name train \
  --initial-strategy-file "$WARMSTARTS_DIR/warmstart_gsm1p5b_attempt3.dfy"
echo "EXIT_SYNTH_GSM1P5B_WARM3=$?"
echo "DONE_GSM1P5B_WARM3 $(date)"
