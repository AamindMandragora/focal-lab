#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON. Successor to
# launch_spider1p5b_fairfix_cold_20260617.sh (that run died inconclusively:
# att1 30.3%, att2 evicted at example 12/300 — no verdict).
#
# WHAT CHANGED vs the dead fairfix run (the framework diff being tested here):
# the IterGen-style helper MENU PRUNE. The usefulness audit
# (saved-results/itergen-style-helper-audit-20260617.md) grepped every recorded/
# accepted win body for `helpers.<Name>(` calls and found only #6
# RollbackConstrainedToComplete (keystone, 4+ clean cold wins) and #5
# RollbackConstrainedSuffix (a21_gsm7b clean win) are used by any clean win; the
# rest of the IterGen-style family is zero-usage clutter on the author's menu.
# User ruling 2026-06-18: prune the 10 unused helpers from the AUTHOR'S MENU (fair
# — a subtraction; gives our method nothing the baselines lack; board-neutral, no
# recorded win used any removed helper). Scope = MENU-ONLY: the names leave
# TOOL_REFERENCE -> leave _ALL_HELPER_NAMES (the bandit/mask universe). The Dafny
# library is NOT edited (#2/#3 stay as hidden internals that kept #5/#6 still call).
#   synthesis/generate/prompts.py md5 dfdba5a69430710ad546f131c2050a0e (was 22a3bfc):
#     10 removed from the menu: RollbackToValidPrefix, RollbackToCompletePrefix,
#     RollbackConstrainedSpan, RollbackAndRegenerate, RollbackAndContinue,
#     SaveLogitsSnapshot, RestoreLogitsSnapshot, RolloutConstrainedWithPenalties,
#     SpeculativeConstrainedRollout, RegenerateUnitOnCheckFailure.
#     4 kept (still bandit-selectable): RollbackConstrainedSuffix,
#     RollbackConstrainedToComplete, DeadEndDetection, RegenerateUnitOnGroundingFailure.
#     Helper universe 72 -> 63 (verified). The earlier fairfix (IterGen promotion
#     removal: stripped When/How-to-use bullets, deleted dead worked example,
#     removed _unit_rewind_hint) is ALSO in this prompts.py (carried forward).
#   TDD guard: synthesis/tests/fairness_cloud/test_helper_menu_prune.py +
#     test_itergen_promotion_removed.py, 12/12 pass on focal py3.12.
#   NOT changed: grammar, grader, splits, helper mask (still ON/bandit), recurrence
#   penalty, the Dafny library (lib md5 7a611864...), eval semantics. No warm start.
#
# WATCH: does the cold author clear the win bar now that the menu is decluttered
# of the 10 zero-usage IterGen-style helpers?
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
OUT=outputs/generated/spider1p5b_prune_cold_20260618
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_prune_cold_20260618 \
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
echo "DONE_SPIDER1P5B_PRUNE_COLD $(date)" | tee -a "$OUT/run.log"
