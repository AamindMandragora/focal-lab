# AGENTS.md — `environment/`

## Scope

**Runtime environment** setup: conda/mxeval install scripts, benchmark split manifests, legacy clone tooling, and tracked patches.

## Rules

- Matrix defaults are **`experiments/splits/gsm_symbolic_crane_proportional_49x49_seed123.json`** for GSM and **`benchmark_splits/spider_dev_proportional.json`** for Spider. Archive other campaign splits under **`experiments/splits/`**.
- **`legacy/`** trees are gitignored; never commit manual edits there without matching **`legacy_patches/`** (see **`legacy/AGENTS.md`**).
- Document new install steps in **`README.md`** here when adding dependencies.

## See also

- **`README.md`** in this folder.
- **`benchmark_splits/README.md`**, **`legacy/AGENTS.md`**, **`legacy_patches/README.md`**.
