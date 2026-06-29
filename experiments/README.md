# Experiments (archived manual runs)

This directory holds **ad-hoc experiment artifacts** that are not part of the core synthesis or baseline pipeline. Nothing here is imported by `synthesis/`, `run_all_tests.py`, or `run_tmux.sh`.

Use it to keep the repository root clean while preserving one-off campaign scripts, warm-start strategy bodies, and non-default benchmark split manifests.

## Layout

| Path | Contents |
|------|----------|
| `scripts/` | Resume, probe, status, and lane supervisor shell scripts from past runs |
| `warmstarts/` | Dafny strategy bodies passed to `--initial-strategy-file` for manual synthesis |
| `splits/` | Extra GSM/Spider split JSONs (seed-specific, 300×300, oracle/probe subsets) |
| `progress.md` | Informal experiment log (optional) |

## Relationship to the main pipeline

- **Matrix / baselines:** `python run_all_tests.py` uses the committed manifests in `environment/benchmark_splits/` only.
- **Synthesis:** `python -m synthesis.run_synthesis` from the repo root.
- **Warm-starts:** copy a body from `warmstarts/` or pass `--initial-strategy-file experiments/warmstarts/<file>.dfy` when launching synthesis manually.

## See also

- Root **`README.md`** for the supported layout and entry points.
- **`AGENTS.md`** in this folder for agent conventions.
