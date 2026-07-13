# Experiment shell scripts

One-off launchers, resume helpers, and status pollers from past manual campaigns. Run from the **repository root** (each script `cd`s there via `lib.sh`).

These are **not** invoked by `run_all_tests.py`. For the supported matrix entry point, use:

```bash
./run_tmux.sh matrix -- --dry-run   # or without --dry-run
```

## Shared setup (`lib.sh`)

Every launcher sources `experiments/scripts/lib.sh`, which defines:

| Variable | Purpose |
|----------|---------|
| `ROOT` | Repository root (auto-detected) |
| `PY` | Python interpreter (`METADECODE_PYTHON` or conda env default) |
| `SPLITS_DIR` | `experiments/splits/` — seed/oracle/probe manifests |
| `ENV_SPLITS_DIR` | `environment/benchmark_splits/` — matrix-default splits |
| `WARMSTARTS_DIR` | `experiments/warmstarts/` — historical strategy bodies referenced by archived scripts |
| `ITERGEN_ROOT` | `legacy/itergen` for native Spider baselines |
| `ITERGEN_NATIVE_PY` | Python for itergen-native eval (falls back to `PY`) |

Override paths with env vars, e.g. `METADECODE_ROOT=/path/to/repo experiments/scripts/gsm1p5b_loopint3.sh`.

## Key helpers

- **`warmstart_retry.sh`** — archived historical SMILES retry launcher. It violates the current cold-start rule and must not be used for new synthesis.
- **`rescore_smiles_unique_valid.py`** — Re-score saved SMILES JSONs on unique-valid / diversity metrics (used by `ab_finish.sh`).

When adapting a script, prefer forwarding flags to `$PY -m synthesis.run_synthesis` or `$PY -m synthesis.evaluate.run_legacy_fixed_strategy` rather than duplicating pipeline logic.
