# AGENTS.md — `environment/benchmark_splits/`

## Scope

**Committed JSON manifests** for fixed GSM-Symbolic and Spider evaluation subsets.

## Rules

- Matrix defaults (`gsm_symbolic_crane_proportional.json`, `spider_dev_proportional.json`) must stay in sync with **`run_all_tests.py`** defaults.
- Non-default or campaign-specific splits belong under **`experiments/splits/`**, not here.
- Changing index sets changes reported metrics; document rationale when updating committed files.

## See also

- **`README.md`** in this folder.
