# GSM warm-extension policy

Date: 2026-07-13

## Approved behavior

```text
Current GSM-14B cycle reaches attempt 40
    success -> held-out evaluation
    failure -> replay attempt 40 + exact prior history -> attempts 41-80

Current GSM Qwen3.5-4B and 9B cycles both reach attempt 40
    either succeeds -> no warm extension for either
    both fail -> independently replay each attempt 40 + exact prior history
                 -> attempts 41-80 for both
```

This is a narrow, user-approved exception to the normal cold-only synthesis
rule. It applies only to `gsm14b`, `gsm-qwen35-4b`, and `gsm-qwen35-9b`, and
only once per cell.

## Billing approval

- AWS account: `887730490125`, recorded UIUC/focal lab account
- Region: `us-east-1`
- Credential source: focal `/home/aadivyar/csd-generation/.env`
- Live STS verification unavailable because focal has no AWS CLI and the
  bearer token does not expose identity through STS
- Approved estimate: about `$20-60` per 40-attempt extension, `$60-180` total

## Safety and state rules

1. An infrastructure interruption before attempt 40 is not a synthesis failure;
   the policy waits for or resumes the current cycle instead.
2. A cycle succeeds only when its synthesis output has a `success_report.json`.
3. A cycle fails only after attempt 40 has a completed evaluation, no success
   report exists, and no matching synthesis process remains alive.
4. Before an extension, rebuild attempts 1-39 from all source logs, recover the
   exact attempt-40 strategy, and replay attempt 40. The replay reconstructs the
   same evaluation-failure refinement prompt before attempt 41 is authored.
5. Each extension has 40 new authored attempts, numbered 41-80. The replay is
   additional bookkeeping and does not consume one of those 40 new attempts.
6. A persistent state file prevents duplicate launches across controller
   restarts. The Qwen3.5 pair decision is recorded atomically before either
   extension launches.
7. Tests and deployment dry-runs must not source `.env`, call Bedrock, or start
   vLLM.

## Implementation and verification

1. Add a small Python policy controller and a shell extension worker under
   `csd-generation/scripts/runtime/`.
2. Write unit tests first for the GSM-14B trigger, the all-or-nothing Qwen3.5
   pair, incomplete/active cycles, successful cycles, history reconstruction,
   attempt numbering, and duplicate prevention.
3. Run tests red, implement, and rerun green.
4. Deploy the tested files to focal `.context`, run syntax checks and a dry-run
   fixture test, then restart only the controller process. Do not interrupt live
   synthesis workers.
