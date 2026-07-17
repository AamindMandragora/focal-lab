# Rendered delimiter API fix and GSM 14B resume

Date: 2026-07-15

## Purpose

Expose the existing verified rendered-text APIs to the synthesis author, remove
the exact-token delimiter assumption from author examples and managed-span
helpers, and resume GSM Qwen2.5-14B from the user-approved attempt-46 checkpoint.

## Billing and experiment settings

- Paid provider account: AWS account `887730490125`, region `us-east-1`.
- Approval: the user previously approved this account and no spending cap; this
  run continues that approved synthesis campaign.
- Credential source checked: focal's live synthesis environment uses
  `AWS_BEARER_TOKEN_BEDROCK` from `/home/aadivyar/csd-generation/.env`. No secret
  value was printed or copied into repository files.
- Author: `us.anthropic.claude-sonnet-4-6` through Bedrock, thinking enabled,
  high effort.
- Evaluator: `Qwen/Qwen2.5-14B-Instruct` through vLLM.
- Data: `gsm_symbolic_crane_proportional_49x49_seed123.json`, `train`, N=49.
- Bars: accuracy `0.5918`, syntax `0.85`.
- User-approved warm exception: replay attempt 46, then author attempts 47-80.

## Changes

- Added `Contains`, `RenderPrefix`, and `RenderedEndsWith` to the author-visible
  core API reference.
- Replaced exact `next == "<<"` checks in all author-facing verified examples
  with a rendered suffix check on the full output after appending the token.
- Added the same rule as a direct system and evaluation-refinement instruction
  after the first newly authored candidate copied the old exact-token pattern
  despite the API documentation. That candidate was stopped before evaluation
  and is excluded from the final lineage.
- Changed `ManagedStep`, `GenerateWithManagedSpan`, and
  `GenerateWithPrefixAndManagedSpan` to use the same full-output suffix check.
- Exposed `DeadEndAvoidingStep`, the only useful callable helper found by the
  audit that was neither visible, deliberately pruned, nor wrapped by a safer
  visible helper.
- Named the four existing logit readers explicitly.
- Audited all 113 owner-qualified callables. Restored four useful general APIs:
  `RollbackAndContinue`, `SaveLogitsSnapshot`, `RestoreLogitsSnapshot`, and
  `SpeculativeConstrainedRollout`. The snapshot pair is locked together so the
  adaptive menu cannot show save without restore.
- Kept seven `CSDHelpers` callables internal with explicit reasons: three are
  lower-level rollback implementations, `RollbackAndRegenerate` and
  `RolloutConstrainedWithPenalties` duplicate safer public controls,
  `FindSubstring` only implements delimiter extraction, and
  `RegenerateUnitOnCheckFailure` accepts an arbitrary allowed-unit list that
  could carry answer information.
- Replaced the missing `ExtractContentExtern` runtime hook with a verified
  recursive substring search and delimiter extractor.
- Added exact speculative-rollout contracts: entry logits are restored,
  completion/EOS flags have exact meanings, and EOS step accounting is explicit.
- Added redacted trace summaries for snapshots, speculative candidates,
  rollback-and-continue, full generation helpers, and penalized rollouts so logs
  retain counts and flags without raw logits or generated sample text. Unknown
  future helper results are redacted by default instead of serialized.
- Updated the recovery manifest to replay attempt 46 from the exact saved
  strategy and load 43 prior evaluated attempts. Attempts 6 and 26 are absent
  because they did not produce evaluation results; the history loader accepts
  evaluated attempts only.

## Verification evidence

- Red run before the first prompt edit: 3 tests failed because the rendered-text
  APIs and exact logit-reader names were absent.
- Red run for the broader fix: 3 tests failed because examples checked only
  `[next]`, managed helpers used exact token equality, and
  `DeadEndAvoidingStep` was hidden.
- Current focused Python suite: `35 passed in 0.09s`.
- Current full Dafny verification on focal: `181 verified, 0 errors`.
- Dafny-to-Python build completed successfully.
- Compiled delimiter extraction runtime sandbox: `4/4` cases passed, including
  missing and unterminated delimiters.
- A 5,017-character extraction case passed after making substring search
  iterative, avoiding Python recursion limits on long generated text.
- Compiled snapshot/speculation sandbox confirmed that mutated logits are
  restored exactly and that a one-token complete candidate reports
  `hitComplete=true`, `hitEos=false`, `stepsUsed=1`.
- Existing prompt-grounding runtime suite on focal: `20 passed in 5.42s`,
  including resemblance behavior and host-method wiring.
- Final independent deployment judge reran the focal grounding suite (`20
  passed`), reran the worktree suite (`35 passed`), inspected the complete
  callable classification and trace fallback, and returned `APPROVED`.
- Focused disposable Dafny sandbox: both
  `RenderedEndsWith([" <<"], "<<")` and
  `RenderedEndsWith([" <", "<"], "<<")` verified.
- Recovery manifest validation loaded `last_clean_attempt=46`, `total_cap=80`,
  the attempt-46 seed, and the before-46 history without error.
- Launch arguments observed on focal:
  `--max-iterations 35 --initial-attempt-offset 45`, which labels the replay as
  attempt 46 and the first new candidate as attempt 47.
- The combined log contains multiple `Attempt 46/80` headers from setup retries:
  one GPU-memory startup failure, one replay stopped before the split-token fix,
  and one replay whose attempt-47 author ignored the documented API. None is a
  reported result. Only the final run launched after the direct system rule is
  part of the result lineage.

## Runtime and logs

- Launcher: `.context/rendered_delimiter_resume_0715/launch_resume_from46.sh`
  on focal.
- Final canonical files were installed on focal after a backup at
  `.context/helper_audit_backup_20260715T0925Z`; local/remote SHA-256 hashes
  matched for the prompt, feedback loop, Dafny library, and trace environment.
- Focal post-deploy checks: `35 passed`; Dafny: `181 verified, 0 errors`.
- GPU 2 was still occupied by a CARS Qwen2.5-7B evaluation (9,300 MiB), so the
  35,000 MiB GSM-14B worker could not start immediately. Transient user service
  `csd-gsm14b-rendered-resume.service` now checks GPU 2 every 15 seconds and
  executes the attempt-46 launcher when usage falls to 3,500 MiB or less.
- Source log: `logs/paid_synth_warmfix_gsm14b_0714_r2.log` on focal.
- The existing focal sync includes `paid_synth_*.log`, and the existing combined
  logger includes that prefix. The combined log showed `Loaded 43 prior evaluated
  attempt(s)` and `Attempt 46/80` from this replay.

To view the combined log locally:

```bash
tail -f "/Users/aadivyar/Documents/Research/Dynamic CSD Gen/local-finalization/csd-generation/logs/combined_data_collection.log"
```

To filter only this GSM 14B run:

```bash
tail -f "/Users/aadivyar/Documents/Research/Dynamic CSD Gen/local-finalization/csd-generation/logs/combined_data_collection.log" | grep --line-buffered paid_synth_warmfix_gsm14b_0714_r2.log
```
