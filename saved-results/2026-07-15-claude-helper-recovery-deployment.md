# Claude helper-recovery deployment

Date: 2026-07-15

## Purpose

Continue every unfinished helper-affected synthesis row on focal without the
Bedrock daily-token blocker. Strategy authoring uses the already-approved
Claude Code Max account `aadivya@fermi.ai`; no API or Bedrock fallback is
allowed.

## Deployed services

- `csd-gsm14b-claude-helper-resume.service` owns GSM Qwen2.5-14B. It replays
  attempt 55 from the complete history through attempt 54, then can author
  attempts 56-80. A permanent claim file prevents a second replay.
- `csd-claude-recovery-queue.service` owns the remaining six rows. It assigns
  work to GPUs that have enough free memory, blocks duplicate row workers,
  runs the frozen held-out command after a synthesis win, and writes a durable
  terminal marker after a held-out result or exhausted train run.
- The older Bedrock recovery service `csd-warm-recovery.service` is disabled
  and inactive.

The six queue rows, in packing order, are:

1. GSM Qwen3.5-2B, replay attempt 24, cap 40.
2. GSM Qwen3.5-4B, replay attempt 24, cap 40.
3. Spider Qwen3.5-4B, replay attempt 39, cap 40.
4. Spider Qwen2.5-7B, replay attempt 22, cap 40.
5. GSM Qwen3.5-9B, replay attempt 20, cap 40.
6. SMILES Qwen3.5-9B isocyanates, replay attempt 30, cap 40.

Spider Qwen3.5-9B is excluded because its helper-affected recovery had already
reached its original 40-attempt cap. GSM Qwen2.5-14B is explicitly rejected by
the six-row launcher because its dedicated one-time service owns it.

## Live verification

At 2026-07-15 13:04 UTC:

- Both services were `active (running)` with `NRestarts=0`.
- GSM14B was evaluating replay attempt 55 on GPU 2.
- GSM Qwen3.5-2B was evaluating replay attempt 24 on GPU 3.
- GSM Qwen3.5-4B was refining replay attempt 24 after a strategy-contract
  failure on GPU 0.
- Spider Qwen3.5-4B was evaluating replay attempt 39 on GPU 1.
- Spider Qwen2.5-7B was evaluating replay attempt 22 on GPU 3.
- GSM Qwen3.5-9B and SMILES Qwen3.5-9B isocyanates remained queued and will be
  assigned when enough GPU memory is free.
- Every synthesis command used `--generation-backend claude`,
  `--generation-model claude-sonnet-4-6`, and
  `--claude-expected-account aadivya@fermi.ai`.
- No synthesis author command used Bedrock.
- The Claude login check reported `loggedIn=true`, `authMethod=firstParty`,
  `subscriptionType=max`, and email `aadivya@fermi.ai`.

GPU snapshot at that check:

```text
GPU 0:    10 MiB / 40960 MiB,  0% utilization
GPU 1: 18549 MiB / 40960 MiB, 32% utilization
GPU 2: 33492 MiB / 40960 MiB, 29% utilization
GPU 3: 35334 MiB / 40960 MiB, 36% utilization
```

GPU 0 was temporarily empty while GSM Qwen3.5-4B was in Claude authoring after
the contract failure; its evaluator process remained assigned to GPU 0.

## Verification results

- Final recovery/provider/default tests on focal: `71 passed, 3 warnings`.
- Wider focal synthesis/provider suite: `117 passed, 3 warnings`.
- Grounding/helper suite: `20 passed, 2 warnings`.
- Independent implementation review: pass after the durable terminal marker,
  explicit GSM14B ownership check, and deployed controller files were added.

The final same-bug search found that the one-time checkpoint builder still
selected the last duplicate attempt even when that block was incomplete. A new
test reproduced the failure, then the builder was changed to select the newest
complete evaluation block while still selecting the newest authored active
strategy. The first focused run failed `1/1`; the final combined run passed
`71/71`.

The service was started and stopped sequentially at 12:54, 12:55, and 12:56
UTC while deployment-review findings were being applied. Those were controlled
pre-final starts, not concurrent controllers. The final queue service started
at 13:00:50 UTC after `launch_queue_cell.py` was deployed at 13:00:12 UTC; it
has zero restarts and one worker per active row. The independent final judge
checked those timestamps and returned **PASS** for the final deployment state.

GSM14B contains two attempt-55 blocks by design. The first belongs to the
provider handoff before the final helper-menu patch. The second is the approved
one-time helper-refresh replay from the same clean boundary. They use distinct
permanent claims, `attempt55.claim` and `attempt55-helper-refresh.claim`, and
only the helper-refresh worker remains active.

The combined-log pipeline needs no new filter. Its remote sync includes
`paid_synth_*.log`, and its follower discovered and copied the new helper-run
logs. Verified output included tagged live lines from
`paid_synth_warmfix_spider-qwen35-4b_0714_r2.log` in:

```text
/Users/aadivyar/Documents/Research/Dynamic CSD Gen/local-finalization/csd-generation/logs/combined_data_collection.log
```

## Reuse and operations

Check both services and GPU use:

```bash
ssh aadivyar@focal "bash -lc 'systemctl --user status csd-gsm14b-claude-helper-resume.service csd-claude-recovery-queue.service --no-pager; nvidia-smi'"
```

Follow the existing combined log locally:

```bash
tail -F "/Users/aadivyar/Documents/Research/Dynamic CSD Gen/local-finalization/csd-generation/logs/combined_data_collection.log"
```

Stop this recovery deployment if needed:

```bash
ssh aadivyar@focal "bash -lc 'systemctl --user disable --now csd-gsm14b-claude-helper-resume.service csd-claude-recovery-queue.service'"
```

The pre-deployment focal backup is:

```text
/home/aadivyar/csd-generation/.context/claude_helper_queue_backup_20260715T1245Z
```
