# H65 GSM-9B stuck-process hotfix

Date: 2026-06-30 Asia/Kolkata / 2026-06-29 UTC

## What this is for

Record the diagnosis and fix for the stale GSM-9B paid Bedrock run.

## Inputs

- Authoritative repo/state: `focal:/home/aadivyar/csd-generation`
- Stale process: PID `284546`
- Old output root: `outputs/generated/synth_gsm_9b_z3fix_seed123train_0628b`
- Approved paid account: AWS account `887730490125`
- Replacement output root: `outputs/generated/synth_gsm_9b_z3fix_seed123train_h65_timeoutguard_20260630`

## Evidence

- PID `284546` was alive but stale:
  - stdout/stderr both pointed to `outputs/generated/synth_gsm_9b_z3fix_seed123train_0628b/run.log`
  - log mtime was frozen at `2026-06-29 01:04:15 UTC`
  - no success/failure/report JSON existed
  - old parent and children were stopped: `[284546, 287538, 287539]`
- Last raw log line was after generation: `Generated 210 tokens in 20.77s`.
- Active evaluator already wrapped scoring in a Python signal timer, so the likely remaining failure mode was native parser/prover work on a pathological GSM expression that Python alarms may not reliably interrupt.
- GSM gold-size audit across train+eval split:
  - max gold expression length: `119`
  - max gold operator count: `22`

## Fix

Added a generous GSM pathological-expression preflight guard before:

1. CRANE/Z3 symbolic equivalence proof in `synthesis/evaluate/evaluator.py`
2. CRANE-faithful final-block syntax parser construction in `synthesis/evaluate/benchmarks/gsm_symbolic/eval_logic.py`

Guard thresholds:

- max expression chars: `512`
- max whitespace tokens: `160`
- max arithmetic/operator chars: `80`
- max repeated digit run: `64`

These thresholds are intentionally far above the observed gold answers, so they should reject only pathological generated outputs.

## Verification

Red test before fix:

```text
2 failed in 0.07s
```

Green focused test after fix:

```text
2 passed in 0.04s
```

Promoted focal-root verification:

```text
7 passed in 0.09s
```

The promoted check included:

- `synthesis/evaluate/test_gsm_pathological_expression_guard.py`
- `synthesis/evaluate/test_metrics.py`
- `py_compile` for the edited evaluator, GSM eval logic, and new test file

## Replacement run

Launched H65 replacement GSM-9B run:

- PID: `464438`
- output: `outputs/generated/synth_gsm_9b_z3fix_seed123train_h65_timeoutguard_20260630`
- log: `outputs/generated/synth_gsm_9b_z3fix_seed123train_h65_timeoutguard_20260630/run.log`
- PID file: `/tmp/csd_h65_logs/h65_gsm9_timeoutguard_20260630.pid`
- launch report: `/tmp/csd_h65_logs/h65_gsm9_timeoutguard_20260630_launch.json`
- command keeps `--max-iterations 40`
- command keeps `--eval-max-seconds-per-example 600`
- scientific settings are the same as the stale GSM-9B run except for the new output id and the evaluator hotfix

Immediate health check:

- old PIDs `284546`, `287538`, and `287539` stopped
- new PID `464438` alive
- log was writing and reached Dafny verification for attempt `1/40`
- launch report recorded `secret_values_written: false`

## Next monitoring step

Watch:

```bash
cat /tmp/csd_h65_logs/h65_gsm9_timeoutguard_20260630.pid
tail -f /home/aadivyar/csd-generation/outputs/generated/synth_gsm_9b_z3fix_seed123train_h65_timeoutguard_20260630/run.log
find /home/aadivyar/csd-generation/outputs/generated/synth_gsm_9b_z3fix_seed123train_h65_timeoutguard_20260630 -maxdepth 3 -type f
```

If H65 reaches a train success, immediately run the held-out GSM-9B re-eval with the fixed strategy using the standard pure re-eval path.
