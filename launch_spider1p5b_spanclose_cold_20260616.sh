#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON, with the SPAN-CLOSE FEEDBACK
# ENRICHMENT (2026-06-16). Successor to launch_spider1p5b_frictionfix_cold_20260616.sh.
#
# WHY: the friction fix was DISPROVEN as the lever (killed @att4, best 26/57; the author
# already used CloseSpanWithinBudget 22x — verification friction was never the bottleneck).
# The real wall is SPAN-NEVER-CLOSES: strategies spend an unconstrained preamble first, then
# hand the close call a depleted budget, so the `<<` span opens but `>>` is never reached
# -> 0% syntax. The author repeated this shape across att2/3/4 because the "span never closed"
# feedback was purely qualitative — it never QUANTIFIED that the preamble ate the budget.
#
# WHAT CHANGED (deployed to focal; this run picks it up, config otherwise byte-identical to the
# friction-fix run so the ONLY difference is the feedback enrichment — clean controlled comparison):
#   feedback_loop.py (md5 ca5e6019): _span_not_closed_hint now reports OBSERVED budget facts on
#   the unterminated outputs — avg tokens spent vs the step budget, how many hit the step ceiling,
#   and how far through the produced text the `<<` span opened (preamble share). Pure measurements,
#   no imperative guidance (fairness-guarded by test). Both call sites pass evaluator.max_steps.
#   NO Dafny change. NOT changed: grammar, grader, splits, prompts, helper mask, recurrence penalty.
#
# WATCH (de-facto probe): does the author BREAK the preamble-heavy shape — preamble-% drops and
# spans start closing — within the first ~5 attempts? If the author keeps opening spans it can't
# close despite now SEEING the budget breakdown, the info gap wasn't the binding constraint.
#
# COLD start (no --initial-strategy-file). Author = Bedrock claude-sonnet-4-6, thinking high (WORK
# cred AWS_BEARER_TOKEN_BEDROCK, us-east-1 — NOT personal). Eval = Qwen2.5-1.5B via vLLM on GPU 2.
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3
OUT=outputs/generated/spider1p5b_spanclose_cold_20260616
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_spanclose_cold_20260616 \
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
echo "DONE_SPIDER1P5B_SPANCLOSE_COLD $(date)" | tee -a "$OUT/run.log"
