# Fixed benchmark splits

Tracked JSON manifests select **fixed, reproducible** example subsets for GSM-Symbolic and Spider evaluation.

Splits use **proportional stratification** over official difficulty labels:

- **GSM-Symbolic (local CRANE JSON folder):** `easy`, `medium`, `hard` from per-file metadata.
- **Spider (dev / HF validation):** `easy`, `medium`, `hard`, `extra` from the vendored Spider evaluator’s `eval_hardness` on gold SQL.

## Matrix defaults (committed here)

| File | Eval split | Size | Notes |
|------|------------|------|-------|
| `gsm_symbolic_crane_proportional.json` | `eval_indices` | 100 | Full local CRANE GSM pool |
| `spider_dev_proportional.json` | `test_indices` / `eval_indices` | 100 test, 50 train | Disjoint subsets of Spider dev |

`run_all_tests.py` passes these paths with `--gsm-split-name eval` and `--spider-split-name eval` (Spider `eval` maps to `test_indices`).

## Regenerating splits

There is no standalone regeneration CLI in the tree. To change the matrix manifests:

1. Update the split-building logic in `synthesis/evaluate/benchmarks/*/dataset.py` (see `write_gsm_proportional_train_eval_split` / `write_spider_proportional_train_test_split` if present), or
2. Produce new JSON offline and replace the committed files above, documenting the change in the commit message.

## Archived splits

Seed-specific, 300×300, oracle, and probe manifests from past campaigns live under **`experiments/splits/`**. They are not used by `run_all_tests.py` unless you pass `--gsm-split-file` / `--spider-split-file` explicitly.
