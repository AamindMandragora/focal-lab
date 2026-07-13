# AGENTS.md — `environment/benchmark_splits/`

## Scope

**Committed JSON manifests** for fixed GSM-Symbolic and Spider evaluation subsets.

## Rules

- The Spider matrix default, **`spider_dev_proportional.json`**, lives here. The GSM matrix default is **`experiments/splits/gsm_symbolic_crane_proportional_49x49_seed123.json`**. Both must stay in sync with **`run_all_tests.py`**.
- Non-default or campaign-specific splits belong under **`experiments/splits/`**, not here.
- Changing index sets changes reported metrics; document rationale when updating committed files.

## See also

- **`README.md`** in this folder.
