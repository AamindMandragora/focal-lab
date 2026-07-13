# AGENTS.md — `experiments/`

## Scope

**Archived manual experiment assets** — not on the synthesis or baseline hot path.

## Rules

- Do **not** import from `experiments/` in `synthesis/`, `run_all_tests.py`, or shared pipeline modules.
- New one-off shell drivers, historical strategy `.dfy` bodies, and non-default split JSONs belong here (or in a subfolder), not at the repository root.
- Scripts should **source `experiments/scripts/lib.sh`** for `ROOT`, `PY`, `SPLITS_DIR`, `ENV_SPLITS_DIR`, and `WARMSTARTS_DIR` instead of hard-coded machine paths.
- Do not use archived strategy bodies to seed new synthesis. They may be used only for pure re-evaluation under the root `AGENTS.md` rule.
- Preserve historical strategy bodies as provenance; do not rewrite them to add synthesis guidance.

## See also

- **`README.md`** in this folder.
- **`scripts/README.md`**, **`warmstarts/README.md`**, **`splits/README.md`**.
