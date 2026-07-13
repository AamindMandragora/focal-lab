# Environment setup

Scripts and manifests for running synthesis, baselines, and the full matrix locally.

## Contents

| Path | Purpose |
|------|---------|
| `install_mxeval_into_env.sh` | Install vendored `mxeval` into the active conda env (required for some SynCode baseline paths) |
| `clone_legacy_csds.sh` | Clone CRANE / IterGen / CARS into `legacy/` and apply `legacy_patches/` |
| `benchmark_splits/` | Committed GSM + Spider manifests for `run_all_tests.py` |
| `legacy/` | Docs and `repos.json` for legacy clones (trees themselves are gitignored under `legacy/`) |
| `legacy_patches/` | Unified diffs applied after clone |
| `vendor/mxeval/` | Vendored mxeval source |

## Conda environment

`run_all_tests.py` uses the active Python environment by default. Override it
with **`VAS_CONDA_ENV`** (legacy alias **`VAS_RDKIT_CONDA_ENV`**). Real runs
check that the selected environment exists and can import RDKit; `--dry-run`
only renders commands and does not require that environment to exist.

Spider SQL tokenization uses NLTK data installed outside the Python wheel. Run
this once in each environment after installing requirements:

```bash
python3 -m nltk.downloader punkt punkt_tab
```

### `mxeval` (Syncode / legacy adapters)

**`amazon-science/mxeval`** is not properly installable from PyPI alone (no **`data/`** next to **`site-packages/mxeval`**, and legacy **`console_scripts`** metadata). After activating your env:

```bash
bash environment/install_mxeval_into_env.sh
```

This clones into **`environment/vendor/mxeval`**, patches **`setup.py`**, copies **`data/`** into **`site-packages`**, and runs **`pip install`**.

### `conda env update`

If you keep a **`vas-eval-environment.yml`** (or similar) for **`conda env update`**, prefer **`pip==24.0`** first in the pip section when pulling **`mxeval`** from Git, or omit **`mxeval`** there and use **`install_mxeval_into_env.sh`** afterward.

If **SciPy** / **transformers** fail with **`CXXABI_1.3.15`** on **`libstdc++`**, **`run_all_tests.py`** prepends **`$CONDA_PREFIX/lib`** to **`LD_LIBRARY_PATH`**; reuse that pattern for other bash drivers.

## Fixed benchmark splits

The matrix uses these disjoint tracked manifests:

- `experiments/splits/gsm_symbolic_crane_proportional_49x49_seed123.json` (49 train / 49 eval, seed 123)
- `environment/benchmark_splits/spider_dev_proportional.json` (300 train / 300 eval, seed 334)

See **`environment/benchmark_splits/README.md`**. Synthesis receives the
`train` split and final scoring receives the disjoint `eval` split.

Additional seed-specific or probe splits are archived under
**`experiments/splits/`** and are not used unless passed explicitly on the CLI.

## Legacy baseline repositories (`legacy/`)

Fixed-strategy runners expect optional upstream clones under **`legacy/CRANE`**, **`legacy/itergen`**, and **`legacy/cars`** (see root **`README.md`**).

```bash
bash environment/clone_legacy_csds.sh
```

Narrative differences between upstream code and this repo’s harness live in **`environment/legacy/DIFFERENCES.md`**. Optional post-clone patches belong under **`environment/legacy_patches/`**. **Any edit under `legacy/*` must be captured as patches** (see **`environment/legacy/AGENTS.md`**). To verify a clean reinstall:

```bash
bash environment/clone_legacy_csds.sh
```

## See also

- **`AGENTS.md`** in this folder.
- Root **`README.md`** for pipeline entry points.
