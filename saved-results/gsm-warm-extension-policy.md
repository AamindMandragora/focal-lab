# GSM warm-extension policy

Date: 2026-07-13 IST

## Purpose

Automatically give selected GSM synthesis failures one additional warm cycle
while preserving the exact final strategy and reconstructing its evaluation
failure before authoring another strategy.

## Approved billing

- AWS account: `887730490125`
- Recorded owner: UIUC/focal lab account, not personal
- Region: `us-east-1`
- Credential source: `/home/aadivyar/csd-generation/.env`
- Live STS verification: unavailable because focal has no AWS CLI and its
  Bedrock bearer token cannot expose identity through STS
- Approved estimate: `$20-60` per extension, up to `$60-180` total
- User confirmation: explicit approval received on 2026-07-13

## Policy

- If the current GSM Qwen2.5-14B cycle completes attempt 40 without a success,
  replay attempt 40 and run 40 new attempts numbered 41-80.
- If both current GSM Qwen3.5-4B and Qwen3.5-9B cycles complete attempt 40
  without success, replay each attempt 40 and run 40 new attempts numbered
  41-80 for each model.
- If either Qwen3.5 cycle succeeds, neither Qwen3.5 extension launches.
- Infrastructure interruptions before attempt 40 do not count as synthesis
  failures.
- The extension is a one-time, explicitly approved exception to the normal
  cold-only rule for these three cells only.

## Exact continuation mechanism

The controller waits for the new failure report produced by the currently
active cycle. It copies the exact attempt-40 strategy into `attempt40.dfy`,
stores every earlier evaluated attempt in `history_before_attempt40.json`, and
launches synthesis with:

```text
--max-iterations 41
--initial-attempt-offset 39
--initial-strategy-file attempt40.dfy
--initial-attempt-history-file history_before_attempt40.json
```

The first CLI iteration re-evaluates attempt 40. That live evaluation rebuilds
the same evaluation-failure refinement prompt before the first new author call,
so the 40 newly authored attempts are numbered 41-80.

## Operations

Controller:

```bash
/apps/conda/aadivyar/envs/csd/bin/python \
  /home/aadivyar/csd-generation/.context/gsm_warm_extension_policy.py
```

Logs and state:

```text
/home/aadivyar/csd-generation/logs/gsm_warm_extension_policy.log
/home/aadivyar/csd-generation/logs/gsm_warm_extension_queue.log
/home/aadivyar/csd-generation/logs/paid_synth_warm_extension_<cell>.log
/home/aadivyar/csd-generation/.context/gsm_warm_extension_policy_state.json
```

## Verification

The focused test suite covers terminal failure detection, active and incomplete
cycles, successful cycles, the GSM-14B trigger, the all-or-nothing Qwen3.5
pair, duplicate prevention, exact attempt-40 seed/history materialization,
stale duplicate report rejection, and the non-billing worker dry run.

```text
10 passed
```
