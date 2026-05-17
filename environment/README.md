# Evaluation conda environment

Default conda prefix for **`run_all_tests.sh`**: **`/apps/conda/advayth2/envs/advayth2`**. Override with **`VAS_CONDA_ENV`** (legacy **`VAS_RDKIT_CONDA_ENV`**).

## `mxeval` (Syncode / legacy adapters)

**`amazon-science/mxeval`** is not properly installable from PyPI alone (no **`data/`** next to **`site-packages/mxeval`**, and legacy **`console_scripts`** metadata). After activating your env:

```bash
bash environment/install_mxeval_into_env.sh
```

This clones into **`environment/vendor/mxeval`** (gitignored), patches **`setup.py`**, copies **`data/`** into **`site-packages`**, and runs **`pip install`**.

## `conda env update`

If you keep a **`vas-eval-environment.yml`** (or similar) for **`conda env update`**, prefer **`pip==24.0`** first in the pip section when pulling **`mxeval`** from Git, or omit **`mxeval`** there and use **`install_mxeval_into_env.sh`** afterward.

If **SciPy** / **transformers** fail with **`CXXABI_1.3.15`** on **`libstdc++`**, **`run_all_tests.sh`** already prepends **`$CONDA_PREFIX/lib`** to **`LD_LIBRARY_PATH`**; reuse that pattern for other bash drivers.

## Legacy baseline repositories (`legacy/`)

Fixed-strategy runners expect optional upstream clones under **`legacy/CRANE`**, **`legacy/itergen`**, and **`legacy/cars`** (see root **`README.md`**).

```bash
bash environment/clone_legacy_csds.sh
```

Narrative differences between upstream code and this repo’s harness live in **`environment/legacy/DIFFERENCES.md`**. Optional post-clone patches belong under **`environment/legacy_patches/`**. **Any edit under `legacy/*` must be captured as patches** (see **`environment/legacy/AGENTS.md`**). To emit a file-level diff against pristine clones:

```bash
python synthesis/scripts/report_legacy_upstream_diff.py --fetch-upstream
```
