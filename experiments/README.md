# Experiments (archived manual runs)

This directory holds **ad-hoc experiment artifacts** that are not part of the core synthesis or baseline pipeline. Nothing here is imported by `synthesis/`, `run_all_tests.py`, or `run_tmux.sh`.

Use it to keep the repository root clean while preserving one-off campaign scripts, historical strategy bodies, and non-default benchmark split manifests.

## Layout

| Path | Contents |
|------|----------|
| `scripts/` | Resume, probe, status, and lane supervisor shell scripts from past runs |
| `warmstarts/` | Historical Dafny strategy bodies retained for provenance and pure re-evaluation |
| `splits/` | Extra GSM/Spider split JSONs (seed-specific, 300×300, oracle/probe subsets) |
| `progress.md` | Informal experiment log (optional) |

## Relationship to the main pipeline

- **Matrix / baselines:** `python3 run_all_tests.py` uses `experiments/splits/gsm_symbolic_crane_proportional_49x49_seed123.json` for GSM and `environment/benchmark_splits/spider_dev_proportional.json` for Spider.
- **Synthesis:** `python3 -m synthesis.run_synthesis` from the repo root.
- **Historical strategies:** do not use these bodies to seed synthesis. They may be passed through `--initial-strategy-file` only for pure re-evaluation with `--max-iterations 1` and zero acceptance bars.

## See also

- Root **`README.md`** for the supported layout and entry points.
- **`AGENTS.md`** in this folder for agent conventions.
