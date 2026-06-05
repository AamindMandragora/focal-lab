# AGENTS.md — `synthesis/evaluate/baselines/`

## Scope

Per-strategy **fixed baseline adapters** that wrap patched legacy code paths.

## Rules

- Add or change strategy behavior in the matching `*.py` adapter and register it in **`registry.py`** (`STRATEGIES`, `ADAPTER_IDS`, `run_baseline_strategy`).
- Prefer harness changes in adapters + **`run_legacy_fixed_strategy.py`** helpers; upstream edits under **`legacy/*`** require patches in **`environment/legacy_patches/<repo>/`** (see **`environment/legacy/AGENTS.md`**).
- SMILES always routes through **`smiles.py`** → **`benchmarks/smiles/pooled_baseline.py`** regardless of strategy.
- GSM/Spider **`crane`** adapter ids accepted for metadecode targets: **`CRANE_ADAPTER_IDS`** in **`registry.py`** (includes legacy `crane_legacy_main` for cached JSONs).
- Do **not** reintroduce ad hoc `~/CRANE` subprocess routing; use **`legacy/CRANE`** + patches.

## See also

- **`README.md`** — adapter table and patch paths.
- **`../run_legacy_fixed_strategy.py`** — shared adapter implementations and matrix CLI entrypoint (`python -m synthesis.evaluate.run_legacy_fixed_strategy`).
