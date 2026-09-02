# Fixed benchmark splits

Tracked JSON manifests select **fixed, reproducible** example subsets for GSM-Symbolic and Spider evaluation.

Splits are built with **proportional stratification** over official difficulty labels:

- **GSM-Symbolic (local CRANE JSON folder):** `easy`, `medium`, `hard` from per-file metadata (rubric / HF match when present).
- **Spider (dev / HF validation):** `easy`, `medium`, `hard`, `extra` from the vendored Spider evaluator’s `eval_hardness` on gold SQL.

Regenerate after changing the CRANE GSM folder or Spider source:

```bash
python -m synthesis.evaluate.benchmarks.write_fixed_benchmark_splits
```

Default manifests:

| File | Eval split | Size | Notes |
|------|------------|------|-------|
| `gsm_symbolic_crane_proportional.json` | `eval_indices` | 100 | Full local CRANE GSM pool |
| `spider_dev_proportional.json` | `test_indices` / `eval_indices` | 100 test, 50 train | Disjoint subsets of Spider dev |

`run_all_tests.py` passes these paths with `--gsm-split-name eval` and `--spider-split-name eval` (Spider `eval` maps to `test_indices`).
