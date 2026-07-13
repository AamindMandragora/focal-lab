# `synthesis/scripts/`

CLI utilities that support the matrix and post-hoc evaluation. They are **not** imported by the core package at import time; invoke them explicitly from the repository root.

## Contents

- **`reevaluate_compiled_csd.py`** — Re-run `Evaluator` on an already-compiled `GeneratedCSD.py` and optionally write a minimal baseline JSON. Used by **`run_all_tests.py`** for Metadecode final eval after synthesis completes.

```bash
python3 -m synthesis.scripts.reevaluate_compiled_csd \
  outputs/generated/<run>/python/GeneratedCSD.py \
  --dataset gsm_symbolic \
  --eval-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-json outputs/baselines/metadecode/...
```

See the module docstring for all flags.

## See also

- **`AGENTS.md`** in this folder for agent constraints when adding scripts.
