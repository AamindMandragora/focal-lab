#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON, with CATEGORIZED VERIFICATION
# REMEDIATION DIAGNOSTICS (2026-06-17). Successor to launch_spider1p5b_helperA_cold_20260617.sh.
#
# WHY: across the last several COLD runs, Dafny VERIFICATION FAILURES dominated the wasted
# attempts (e.g. 4 of 7 in the span-close run; high in helperA). Characterizing all 37 verify
# errors across 6 logs showed the PROGRESS postcondition is the dominant bucket (24/37), not the
# cost/budget bug (2/37). The verification refinement prompt already received a parsed/categorized
# "Structured verifier analysis" block (obligation_kind, failing code, the cited contract excerpt)
# but never told the author HOW to satisfy the obligation that failed — unlike the eval refinement
# prompt, which has a categorized "## How to revise" block.
#
# WHAT CHANGED (deployed + verified on focal; config otherwise byte-identical to the helperA run so
# the ONLY framework difference is the new remediation layer — clean controlled comparison):
#   synthesis/verify/verifier.py (md5 62ecb1ceb702ccbf10769d8851dd7a2f): added _remediation_for() +
#   _cited_contract_line(); get_structured_feedback() now appends a mechanism-level "Remediation:"
#   line per diagnostic, sub-typed within the postcondition bucket by the cited `ensures` clause
#   (progress / length / cost), plus precondition / invariant / decreases hints. Local TDD 8/8 green;
#   fairness-guarded (no task/strategy/baseline tokens — mechanism only, mirrors the accepted eval
#   "## How to revise" block). NO prompt-template change (the {structured_feedback_block} slot was
#   already wired). NOT changed: grammar, grader, splits, prompts, helper mask, recurrence penalty,
#   the GenerateWithPrefixAndManagedSpan scaffold (still a prunable helper, bandit decides). No warm start.
#
# WATCH (de-facto probe — the user's exact question): does the verify-fail RATE per attempt drop vs
# the 4/7 of the span-close run / the high rate in helperA, and does the author recover from a verify
# failure in FEWER refinement iterations now that each failure carries a targeted remediation?
#
# COLD start (no --initial-strategy-file). Author = Bedrock claude-sonnet-4-6, thinking high (WORK
# cred AWS_BEARER_TOKEN_BEDROCK, us-east-1 — NOT personal). Eval = Qwen2.5-1.5B via vLLM on GPU 1
# (orphans reaped 2026-06-17, ~38GB free), 0.20 util.
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3
OUT=outputs/generated/spider1p5b_verbdiag_cold_20260617
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_verbdiag_cold_20260617 \
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
echo "DONE_SPIDER1P5B_VERBDIAG_COLD $(date)" | tee -a "$OUT/run.log"
