#!/bin/bash
# GSM-1.5B restart #3 after warm14 exhaustion (expected NO_ACCEPT, anchor stuck at 38.8%/57.1%).
# Audit of warm14 attempts 12/14/15/16 (agent a213ab611a058472d): the '<<' intercept WAS
# implemented (EnterObservedConstrainedSpan on next=="<<") but only caught the FIRST span;
# the model emits 4-6 '<<' spans across free generation, so later spans pass unmasked and
# 48/49 examples fail on unterminated/invalid spans. Short free phase (30) costs accuracy
# (28.6%); the fix is to LOOP the intercept for the entire generation, not shorten the free phase.
# Warm start: attempt-14 body (38.8%/57.1% anchor), already on focal.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
NAME="gsm1p5b_seed123_loopint_20260611"

echo "=== GSM1P5B-LOOPINT START $(date) (warm att14, bars 0.33acc/0.90syn) ==="
$PY -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters. CSD mechanics that matter: the eval scores EVERY visible << >> span against the grammar, and the model emits '<<' MANY times over one solution - routing only the first occurrence into the constrained path leaves every later span unmasked and unterminated, which is the dominant failure. The strategy's main loop must therefore ALTERNATE for the entire generation: an unconstrained chunk that stops at the next '<<', then a parser-guided constrained span closed at a grammar-complete point, then back to unconstrained - repeating until the step budget or EOS, so EVERY span the model opens is parser-guided. Do not rely on a single early forced span, and enforce the constrained-span token budget inside the loop so no opened span is left unclosed." \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name "$NAME" --output-dir "outputs/generated/$NAME" \
  --min-accuracy 0.33 --min-syntax-rate 0.90 \
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
  --initial-strategy-file "$WARMSTARTS_DIR/warmstart_gsm1p5b_attempt14.dfy"
echo "EXIT_SYNTH_GSM1P5B_LOOPINT=$?"
echo "DONE_GSM1P5B_LOOPINT $(date)"
