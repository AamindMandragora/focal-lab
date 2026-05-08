# Outputs

All runtime artifacts are stored under this directory.

- `generated/`: synthesized CSD run outputs (`dafny/`, `python/`, `results/`).
- `baselines/`: minimal baseline JSON snapshots.

Baseline JSON policy:

- store only `accuracy`, `syntax_rate`, and per-question generated answers.

Matrix helper:

- Run `./run_all_tests.sh` from repo root to execute the default full matrix and ablations.
