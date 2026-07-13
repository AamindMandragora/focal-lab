# Non-default benchmark splits

Extra GSM-Symbolic and Spider index manifests used by scripts under `experiments/scripts/`. Examples include seed-specific 49×49 GSM pools, 300×300 Spider subsets, and oracle/probe structural splits.

The matrix uses these tracked defaults:

- `experiments/splits/gsm_symbolic_crane_proportional_49x49_seed123.json`
- `environment/benchmark_splits/spider_dev_proportional.json`

Pass other archived splits explicitly when re-running a campaign, e.g.:

```bash
--gsm-split-file experiments/splits/gsm_symbolic_crane_proportional_49x49_seed123.json \
--gsm-split-name eval
```

JSON metadata fields such as `crane_dir` / `spider_dir` are informational; runtime loading uses repo defaults (`legacy/CRANE`, `SPIDER_DATA_DIR`, etc.) unless overridden on the CLI.
