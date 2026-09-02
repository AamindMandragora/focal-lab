#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON. Successor to
# launch_spider1p5b_prune_cold_20260618.sh.
#
# WHAT CHANGED vs the prune_cold run (the framework diff being tested here):
# the Pattern A TEMPLATE RESTRUCTURE. The goal #36 verify-failure audit found
# ~90% (43/48) of synthesis verification failures were the PROGRESS postcondition
# failing on author no-op early-return passthroughs. Fix (fair — pipeline template
# scaffolding, not grammar/grader/split/eval-semantics): the focal LIVE template
# synthesis/verify/library/GeneratedCSD.dfy now splices the author body into a new
# sub-method AuthorBody(...) that carries every postcondition EXCEPT PROGRESS and
# pre-inits the 4 out-params; MyCSDStrategy(...) keeps its FULL unchanged 6-ensures
# contract and re-establishes PROGRESS via `if maxSteps > 0 && cost <= 0 { cost := 1; }`.
# No ensures weakened. prompts.py NOT changed and does not embed the template, so
# the author never sees AuthorBody — controlled-study inputs are byte-identical.
#   Deployed template md5 41c272a1 (orig backed up *.bak_preAuthorBody_20260619).
#   TDD: local RED (current template + passthrough fails PROGRESS rc=4) -> GREEN
#   (restructured passes passthrough/CRANE/empty). On focal: all 3 bodies dafny
#   verify rc=0 AND dafny build --target:py "4 verified, 0 errors".
#   NOT changed: grammar, grader, splits, helper mask (still ON/bandit), recurrence
#   penalty, eval semantics, the menu prune (still applied), the Dafny library
#   (lib md5 3c088b24). No warm start.
#
# WATCH: does the cold author clear the win bar now that Pattern A verify-friction
# (the dominant rejection cause) is removed by the template?
#
# COLD start (no --initial-strategy-file). Author = Bedrock claude-sonnet-4-6,
# thinking high (WORK cred AWS_BEARER_TOKEN_BEDROCK from synthesis/.env, us-east-1
# — NOT personal; approved by user 2026-06-19). Eval = Qwen2.5-1.5B via vLLM on
# GPU 1, 0.20 util.
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3
OUT=outputs/generated/spider1p5b_authorbody_cold_20260619
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_authorbody_cold_20260619 \
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
echo "DONE_SPIDER1P5B_AUTHORBODY_COLD $(date)" | tee -a "$OUT/run.log"
