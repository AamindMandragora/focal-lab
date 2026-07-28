# Warm task-context recovery approval

Date: 2026-07-14

## Purpose

Recover eight synthesis searches whose continuation prompts lost the task description because of the warm-resume bug.

## User-approved exception

The user explicitly overrode the project's cold-only synthesis rule for these eight recovery runs. Each run may replay its last verified clean strategy and continue from the following attempt. This exception applies only to the manifest in `saved-results/2026-07-14-warm-task-recovery-manifest.json`.

## Billing

- AWS account: `887730490125`
- Region: `us-east-1`
- Credential source: `/home/aadivyar/csd-generation/.env` on focal, variable `AWS_BEARER_TOKEN_BEDROCK`
- Account verification limitation: the bearer token does not support an STS identity query and the focal shell has no AWS CLI. The account is the previously recorded, user-approved account for this campaign.
- Approval: the user previously authorized spending as much as necessary with no maximum cap, and explicitly requested this eight-run recovery queue.

## Cost estimate

The prior approved estimate was `$20-60` per 40-attempt Sonnet-4.6 thinking-high synthesis run. The recovery manifest contains 219 replay-plus-new attempts, or 5.475 equivalent 40-attempt runs:

```text
5.475 × $20-60 = approximately $110-330
```

Actual cost can be lower if runs accept early. Held-out evaluations use local focal models and do not call Bedrock.

## Scope

- GSM Qwen3.5-2B from clean attempt 10
- GSM Qwen3.5-4B from clean attempt 10
- GSM Qwen3.5-9B from clean attempt 3
- GSM Qwen2.5-14B from clean attempt 33 through cap 80
- Spider Qwen3.5-4B from clean attempt 39
- Spider Qwen3.5-9B from clean attempt 37
- SMILES Qwen3.5-9B isocyanates from clean attempt 1
- Archived Spider Qwen2.5-7B from clean attempt 8

## Launch record

- Final controller launched at `2026-07-14 12:04:47 UTC` on focal.
- Controller PID: `3418901`.
- Controller log: `logs/paid_synth_warmfix_recovery_queue_driver_r2.log`.
- Combined local log: `logs/combined_data_collection.log` under the local-finalization checkout.
- First dispatch: GSM Qwen2.5-14B on GPU 3; GSM Qwen3.5-9B on GPU 0; Spider Qwen3.5-9B on GPU 1; SMILES Qwen3.5-9B isocyanates on GPU 2.
- GSM Qwen3.5-2B, GSM Qwen3.5-4B, Spider Qwen3.5-4B, and archived Spider Qwen2.5-7B stay in the same greedy queue until enough GPU memory is available.
- An initial controller launched at `11:55:37 UTC` was stopped after a live packing check showed that the GSM 2B job's 40% vLLM request required 16,384 MiB, more than its stale 13,000 MiB manifest estimate. No evaluation result from that failed start was retained. The scheduler now reserves the larger of the manifest estimate and the requested vLLM fraction of physical GPU memory; the final controller was launched only after the targeted test and full focal suite passed.

## Verification

- Red test reproduced the unsafe 13,000 MiB packing decision before the scheduler fix.
- Final focused focal suite: `35 passed, 2 warnings in 4.12s`.
- Full manifest dry run recovered all eight exact checkpoints and returned success.
- Live r2 scan found one controller, four active r2 workers, zero non-r2 synthesis workers, and no traceback, memory error, HTTP 429, `Unknown task`, or nonzero worker status.
- The local combined logger and Luna/WhatsApp supervisor both discovered the r2 controller and worker logs.
- Independent judge result: PASS. It verified the eight attempt ranges, checkpoint provenance, task-context initialization, GPU packing, duplicate locks, automatic held-out wiring, and combined-log visibility.

## Persistent focal automation

The user approved automatic recovery on focal and confirmed that the existing AWS Bedrock approval and focal `.env` credential apply. No credential value is stored in code, the manifest, or the systemd unit.

- User service: `csd-warm-recovery.service`
- Installed unit: `/home/aadivyar/.config/systemd/user/csd-warm-recovery.service`
- Supervisor: `scripts/runtime/supervise_warm_task_recovery.py`
- State: `.context/warm_task_recovery_0714/supervisor_state.json`
- Pending manifest: `.context/warm_task_recovery_0714/pending_manifest.json`
- Supervisor log: `logs/paid_synth_warmfix_supervisor.log`
- Retry interval after an unfinished controller exit: 3,600 seconds
- Restart behavior: the enabled user service restarts on failure and systemd terminates its complete worker process group before restart.
- Takeover behavior: the service first adopts controller PID `3418901`; it does not interrupt the live r2 workers. If that controller exits, terminal failure reports are skipped, successful synthesis rows resume directly at held-out evaluation, and only rows without terminal reports restart from their approved clean checkpoints.

Deployment verification:

- Focal focused suite: `43 passed, 2 warnings in 4.55s`.
- Fake crash/restart integration cycle passed.
- `systemd-analyze --user verify` passed.
- Live supervisor restart retained controller PID `3418901` before and after the restart.
- Service status after restart: `enabled` and `active`.
- Combined logger discovered `paid_synth_warmfix_supervisor.log`.
- Exact-controller adoption requires both the controller script and the resolved approved manifest path. A live probe returned PID `3418901` for the approved manifest and `None` for a different manifest.
- Stale-worker cleanup follows the exact matched synthesis process trees recursively. A live read-only probe matched all four active vLLM engine descendants and excluded unrelated process trees.
- Independent persistence audit: PASS after verifying exact-manifest adoption, all vLLM descendants, unchanged worker start times, one controller, one supervisor, no duplicates, deployment hash parity, hourly retry behavior, combined/Luna visibility, and absence of the Bedrock token outside focal's `.env`.

## Codex catastrophic-incident recovery

The user also approved a separate focal monitor that can stop the exact recovery queue, ask the logged-in Codex CLI to repair an operational bug in an isolated source copy, independently verify the change, and relaunch the existing recovery service. This does not expand the eight-row warm-start exception.

- Service: `csd-codex-incident-monitor.service`
- Monitor: `scripts/runtime/incident_repair/monitor.py`
- Incident log: `logs/paid_synth_codex_incident_monitor.log`
- State and evidence: `.context/codex_incident_monitor/`
- Codex login: focal's existing ChatGPT login; no API key is copied into the service.
- Model: `gpt-5.5`, the strongest supported default exposed by focal's ChatGPT-authenticated Codex CLI during the live check. The requested `gpt-5.6-sol` returned HTTP 400 as unsupported.
- Codex sandbox: write access only to a temporary source snapshot.
- Credentials removed from the repair process: all `AWS_*`, `BEDROCK_*`, and `*_API_KEY` variables.
- Protected from autonomous edits: grammars, graders, scorers, equivalence code, benchmark splits, datasets, model choices, score bars, recipes, warm/cold policy, results matrix, recovery manifest, and `.env` files.
- Relaunch gate: structured Codex success, no protected/deleted/out-of-scope files, focused tests passing in the snapshot, and the same tests passing after deployment.
- Failure behavior: if any gate fails, the synthesis recovery service stays stopped and the paid-synthesis log records `REPAIR_BLOCKED` for WhatsApp/Luna visibility.

Verification record:

- Monitor and supervisor safety suite: `21 passed` on focal after final cleanup, including a fake `Unknown task` repair/deploy/relaunch cycle and rollback when relaunch fails.
- Full focused focal suite after the independent audit fixes: `55 passed, 2 warnings in 4.22s`.
- Live structured-output Codex smoke: `gpt-5.5` returned schema-valid JSON without editing files.
- Independent safety review initially found six gaps: nested credential copying, checkout-insensitive process matching, broad scientific-file write access, stale replay for logs discovered later, weak structured proof checks, and missing rollback after restart failure. Each received a focused test and code fix. The second independent review returned `PASS`.
