#!/bin/bash
# GSM-1.5B loopint4: pivot to SUPPRESS-THEN-FORCE after loopint3 plateaued at 44.9/59.2.
# Attempt-7 failure analysis (agent a02ce38f): 20/20 syntax failures are terminated spans with
# SYMBOLIC content written free-phase; intercept fired 2/49 (token-equality trigger again);
# intercepting intermediate spans burns budget. New mechanism: mask delimiter-opening token in
# free phase (no natural spans), then force ONE constrained final-answer span with reserve.
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
NAME="gsm1p5b_seed123_loopint4_20260611"

echo "=== GSM1P5B-LOOPINT4 START $(date) (warm loopint3-att7, suppress-then-force, bars 0.33acc/0.90syn) ==="
$PY -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping the final answer inside << >> delimiters. CSD mechanics that matter: the eval scores EVERY visible << >> span against the grammar, and this weak model, left free, writes several << >> spans containing symbolic variable names the grammar rejects - reactive interception of those spans has repeatedly failed (token-level triggers miss split delimiters, and intercepting an intermediate reasoning span burns the budget mid-sentence). The reliable mechanism is SUPPRESS-THEN-FORCE: (1) during the free reasoning phase, mask out the token(s) that begin the span-opening delimiter so the model CANNOT open a span on its own - it reasons in plain text instead; (2) after the reasoning phase, FORCE exactly one constrained span for the final answer - append the opening delimiter directly, drive the content through the parser-guided constrained path, and close at a grammar-complete point - reserving enough of the step budget for this span so it always completes; (3) never exit the span without emitting its closing delimiter (roll back to the last grammar-complete prefix and close there if generation stalls). One forced, fully parser-guided span per example satisfies the delimiter requirement with grammar-valid content." \
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
  --no-adaptive-helper-mask \
  --refinement-beam-size 2 \
  --gsm-split-file "$SPLITS_DIR/gsm_symbolic_crane_proportional_49x49_seed123.json" \
  --gsm-split-name train \
  --initial-strategy-file "$WARMSTARTS_DIR/warmstart_gsm1p5b_loopint3_att7.dfy"
echo "EXIT_SYNTH_GSM1P5B_LOOPINT4=$?"
echo "DONE_GSM1P5B_LOOPINT4 $(date)"
