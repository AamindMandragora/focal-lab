#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON. Successor to
# launch_spider1p5b_managedstep_cold_20260617.sh (killed 2026-06-17).
#
# WHAT CHANGED vs the killed run (the ONLY framework difference — clean controlled
# comparison): the IterGen-type-tool fairness promotion was removed from the
# synthesis prompt + feedback (deployed + verified on focal 2026-06-17):
#   synthesis/generate/prompts.py (md5 22a3bfc72b4693f065821e4e5a1f039b):
#     - the two grounding-helper menu bullets (RegenerateUnitOnCheckFailure /
#       RegenerateUnitOnGroundingFailure) stripped to mechanism-only (KEPT
#       signature/Role/Mechanics/Cost/Control profile; REMOVED When-to-use /
#       How-to-use / Example-call-shape / Suggested-starting-values).
#     - deleted the DEAD "Grounded-and-closed" worked example (it never matched a
#       _VERIFIED_EXAMPLE_PREFIXES entry, so it was never injected) + its dangling
#       prefix. Helper universe unchanged = 72; both helpers still selectable.
#   synthesis/evaluate/feedback_loop.py (md5 feb9fb5fc25521228842dc058b60d61d):
#     - removed _unit_rewind_hint (the feedback nudge that named
#       RegenerateUnitOnCheckFailure and told the author to adopt it). Sibling
#       general "force <</reach >>" CSD-mechanism hints kept (user-ruled fair).
#   NOT changed: grammar, grader, splits, helper mask (still ON/bandit), recurrence
#   penalty, the Dafny library (ManagedStep still present, lib md5 7a611864...),
#   eval semantics. Pre-existing general guidance (prompts.py ~912-923, ~2192) left
#   AS-IS per user ruling 2026-06-17 ("leave it, re-run now" — task-agnostic = fair).
#   No warm start.
#
# WATCH: does the cold author clear the win bar now that the IterGen promotion is
# gone (i.e. was the promotion masking a real signal, or irrelevant)?
#
# COLD start (no --initial-strategy-file). Author = Bedrock claude-sonnet-4-6,
# thinking high (WORK cred AWS_BEARER_TOKEN_BEDROCK from synthesis/.env, us-east-1
# — NOT personal). Eval = Qwen2.5-1.5B via vLLM on GPU 1, 0.20 util.
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3
OUT=outputs/generated/spider1p5b_fairfix_cold_20260617
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_fairfix_cold_20260617 \
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
echo "DONE_SPIDER1P5B_FAIRFIX_COLD $(date)" | tee -a "$OUT/run.log"
