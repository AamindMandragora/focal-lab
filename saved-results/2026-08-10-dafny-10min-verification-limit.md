# Dafny 10-minute verification time limit (verify + compile)

- Date: 2026-08-10
- What for: Raise shared `--verification-time-limit` so `dafny build` no longer dies on Dafny’s default 30s after `dafny verify` already passed under a higher cap.
- Worktree: `csd-generation-wt-dafny-10min-vlimit` branch `dafny-10min-verification-limit`
- Merged into: focal `full-baseline-campaign-20260803` (data-collection launch tree)

## Change

| Knob | Before | After |
|------|--------|-------|
| `--verification-time-limit` (verify) | 120s | **600s (10 min)** |
| `--verification-time-limit` (compile/`dafny build`) | unset → Dafny default **30s** | **600s (same)** |
| Verify process timeout | 180s | **900s** |
| Compile process timeout | 120s (compiler default) | **900s** |

Constants: `synthesis/run_constants.py`. Builders: `synthesis/verify/tooling.py`.

## Tests

`python3 -m pytest tests/test_dafny_verification_time_limit.py -v` → 3 passed
