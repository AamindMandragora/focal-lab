# AGENTS.md — `experiments/`

## Scope

**Archived manual experiment assets** — not on the synthesis or baseline hot path.

## Rules

- Do **not** import from `experiments/` in `synthesis/`, `run_all_tests.py`, or shared pipeline modules.
- New one-off shell drivers, warm-start `.dfy` bodies, and non-default split JSONs belong here (or in a subfolder), not at the repository root.
- Scripts should **source `experiments/scripts/lib.sh`** for `ROOT`, `PY`, `SPLITS_DIR`, `ENV_SPLITS_DIR`, and `WARMSTARTS_DIR` instead of hard-coded machine paths.
- Do not add synthesis **strategy guidance** to warm-start bodies beyond what a normal CSD strategy would contain.

## See also

- **`README.md`** in this folder.
- **`scripts/README.md`**, **`warmstarts/README.md`**, **`splits/README.md`**.
