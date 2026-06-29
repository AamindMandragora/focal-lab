#!/bin/bash
# GSM-1.5B loopint2: restart after loopint plateaued (anchor att9 40.8/57.1 unbeaten through att15).
# Audit: (1) reactive '<<' trigger used token equality -> fired 1/49 (tokenizer splits '<<'); (2) rejected
# span content rolls back to empty and '>>' never emitted -> unterminated. Guidance now names both mechanics.
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
export CUDA_VISIBLE_DEVICES=1  # GPU 0 blocked: orphaned engine (12.7GB, ours, needs user-approved kill) + labmate job (18.5GB)
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
# survive Bedrock daily-quota windows: retries are read per-call from env
export CSD_API_MAX_RETRIES=10
NAME="gsm1p5b_seed123_loopint2_20260611"

echo "=== GSM1P5B-LOOPINT2 START $(date) (warm loopint-att9, bars 0.33acc/0.90syn) ==="
$PY -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters. CSD mechanics that matter: the eval scores EVERY visible << >> span against the grammar, and the model emits '<<' MANY times over one solution - routing only the first occurrence into the constrained path leaves every later span unmasked and unterminated, which is the dominant failure. The strategy's main loop must therefore ALTERNATE for the entire generation: an unconstrained chunk that stops at the next '<<', then a parser-guided constrained span closed at a grammar-complete point, then back to unconstrained - repeating until the step budget or EOS, so EVERY span the model opens is parser-guided. Do not rely on a single early forced span. TWO MECHANICS ARE CRITICAL: (1) detect the '<<' delimiter at the STRING level - an unconstrained chunk that stops when the generated text reaches '<<' - never by comparing a single sampled token to '<<', because the tokenizer can split the delimiter across two tokens so a token-equality trigger may never fire even when the text contains many '<<'; (2) never exit a constrained span without emitting its closing delimiter - check grammar-completeness after every appended token and close at the FIRST complete point, and if generation stalls while the span is incomplete, roll back to the last grammar-complete prefix and close there, deterministically appending grammar-valid tokens to reach a complete point if none exists - an opened '<<' without its '>>' is an automatic syntax failure." \
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
  --initial-strategy-file "$WARMSTARTS_DIR/warmstart_gsm1p5b_loopint_att9.dfy"
echo "EXIT_SYNTH_GSM1P5B_LOOPINT2=$?"
echo "DONE_GSM1P5B_LOOPINT2 $(date)"
