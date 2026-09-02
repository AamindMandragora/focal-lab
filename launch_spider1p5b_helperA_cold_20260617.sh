#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON, with the VERIFY-FRICTION HELPER
# ABSTRACTION (2026-06-17). Successor to launch_spider1p5b_spanclose_cold_20260616.sh.
#
# WHY: the span-close FEEDBACK run was killed @att7 — the feedback lever was DORMANT (Att1 already
# closed spans at 78% syntax so the span-not-closed hint never fired) AND verification failures
# DOMINATED (4 of 7 attempts died at the Dafny prover). Root cause (investigated): the author keeps
# a local `steps` counter, calls helpers that ALSO auto-increment the `cost` field, then writes
# `cost := steps` at the end — a dual-counter scheme Dafny can't prove (steps double-counts, EOS
# break paths leave it untieable to the field). The author re-derives a 3-phase loop (bounded
# preamble -> constrained loop -> close) by hand and botches the budget proof every time.
#
# WHAT CHANGED (deployed + verified on focal; config otherwise byte-identical to the span-close run):
#   1. VerifiedAgentSynthesis.dfy (re-verified 173 verified / 0 errors): added
#      GenerateWithPrefixAndManagedSpan — a VERIFIED one-call scaffold (bounded unconstrained
#      preamble of prefixBudget steps, then a managed constrained span that closes eagerly). A single
#      unified step counter discharges length+cost+progress BY CONSTRUCTION, so a strategy is one call
#      + `cost := helpers.cost` with NO hand-rolled bookkeeping. Proven end-to-end: a one-call strategy
#      through the real template verifies 2/0. The bounded preamble is ALSO the span-closing fix.
#   2. feedback_loop.py (md5 d51b4b8b): GenerateWithPrefixAndManagedSpan is a PRUNABLE helper, same tier
#      as every other full-loop scaffold (user ruling 2026-06-17: helpers prunable by default; only
#      primitives/introspection are non-prunable; hand-promoting a good scaffold nudges the author and
#      defeats the mask). The bandit decides whether the author sees it. The span-close feedback
#      enrichment is also still in this file. |NON_PRUNABLE|=41 |PRUNABLE|=59 (the 6 scaffolds moved in).
#   NOT changed: grammar, grader, splits, prompts (no strategy guidance), eval semantics, helper mask,
#   recurrence penalty. No warm start.
#
# WATCH (de-facto probe): (a) verify-fail rate — target WELL BELOW the 4/7 of the last run; and
# (b) does the author USE GenerateWithPrefixAndManagedSpan (grep the strategies)? If verify-fails
# stay high AND the author ignores the helper, the abstraction isn't being adopted -> reassess.
#
# COLD start. Author = Bedrock claude-sonnet-4-6, thinking high (WORK cred AWS_BEARER_TOKEN_BEDROCK,
# us-east-1 — NOT personal). Eval = Qwen2.5-1.5B via vLLM on GPU 1, 0.20 util (GPU2 filled up — a
# 19.6GB job landed there at 97% util; GPU1 has ~13GB free).
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3
OUT=outputs/generated/spider1p5b_helperA_cold_20260617
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_helperA_cold_20260617 \
  --min-accuracy 0.55 --min-syntax-rate 0.85 \
  --eval-sample-size 300 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 300 --eval-min-examples-before-threshold-stop 300 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  > "$OUT/run.log" 2>&1
echo "SYNTH_EXIT=$?" | tee -a "$OUT/run.log"
echo "DONE_SPIDER1P5B_HELPERA_COLD $(date)" | tee -a "$OUT/run.log"
