# AGENTS.md — `environment/`

## Scope

**Runtime environment** setup: conda/mxeval install scripts, benchmark split manifests, legacy clone tooling, and tracked patches.

## Rules

- Matrix-default split JSONs live in **`benchmark_splits/`** only (`gsm_symbolic_crane_proportional.json`, `spider_dev_proportional.json`). Archive non-default splits under **`experiments/splits/`**.
- **`legacy/`** trees are gitignored; never commit manual edits there without matching **`legacy_patches/`** (see **`legacy/AGENTS.md`**).
- Document new install steps in **`README.md`** here when adding dependencies.

## See also

- **`README.md`** in this folder.
- **`benchmark_splits/README.md`**, **`legacy/AGENTS.md`**, **`legacy_patches/README.md`**.
