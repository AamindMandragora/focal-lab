# Fixed benchmark splits

Tracked JSON manifests select **fixed, reproducible** example subsets for GSM-Symbolic and Spider evaluation.

Splits use **proportional stratification** over official difficulty labels:

- **GSM-Symbolic (local CRANE JSON folder):** `easy`, `medium`, `hard` from per-file metadata.
- **Spider (dev / HF validation):** `easy`, `medium`, `hard`, `extra` from the vendored Spider evaluator’s `eval_hardness` on gold SQL.

## Tracked splits

| File | Train / eval size | Seed | Notes |
|------|-------------------|------|-------|
| `experiments/splits/gsm_symbolic_crane_proportional_49x49_seed123.json` | 49 / 49 | 123 | Disjoint matrix default |
| `gsm_symbolic_crane_proportional.json` | 0 / 100 | 334 | Evaluation-only historical manifest; not valid for synthesis |
| `spider_dev_proportional.json` | 300 / 300 | 334 | Disjoint Spider matrix default |

`run_all_tests.py` uses the seed123 GSM split above and the seed334 Spider
split by default. Synthesis receives `train_indices`; final evaluation receives
the disjoint `eval_indices` (Spider `eval` maps to `test_indices`).

## Regenerating splits

There is no standalone regeneration CLI in the tree. To change the matrix manifests:

1. Update the split-building logic in `synthesis/evaluate/benchmarks/*/dataset.py` (see `write_gsm_proportional_train_eval_split` / `write_spider_proportional_train_test_split` if present), or
2. Produce new JSON offline and replace the committed files above, documenting the change in the commit message.

## Archived splits

Additional seed-specific, oracle, and probe manifests from past campaigns live
under **`experiments/splits/`**. Apart from the default seed123 GSM manifest,
they are used only when passed with `--gsm-split-file` or `--spider-split-file`.
