# Legacy baseline codebases (local only)

These directories are **not** committed: they are large upstream snapshots used by
`python -m synthesis.evaluate.run_legacy_fixed_strategy` for CRANE / IterGen /
CARS baselines (GCD uses vendored SynCode under `synthesis/evaluate/syncode/`).

## Install

From the repository root:

```bash
bash environment/clone_legacy_csds.sh
```

Override upstream URLs or refs with environment variables (see
`environment/clone_legacy/repos.json` and the clone script header).

## Expected layout

| Path | Used for |
|------|-----------|
| `legacy/CRANE/` | `unconstrained`, `crane` strategies (`legacy/CRANE/src/main.py`) |
| `legacy/itergen/` | `itergen` strategy (`from itergen.main import IterGen`) |
| `legacy/cars/` | `cars` strategy (`from cars.lib import ConstrainedModel`) |

## How this repo differs from upstream

See **`environment/legacy/DIFFERENCES.md`** for harness-side behavior (cache
paths, vendored SynCode, grammar tightening, GSM normalization).

To produce a **file-level** summary against freshly cloned upstream trees:

```bash
python synthesis/scripts/report_legacy_upstream_diff.py --fetch-upstream
```

Or compare against your own mirror directory:

```bash
python synthesis/scripts/report_legacy_upstream_diff.py --upstream-base /path/with/CRANE_itergen_cars
```

Optional unified patches applied after clone live under **`environment/legacy_patches/`**
(see that folder’s README). CRANE apply order ends with **`030-vas-smiles-prompt-state-grammar`**
(SMILES **`SmilesPromptState`** + base-grammar scoring fallback). **Policy:** any manual change under **`legacy/*`** must be
mirrored there — see **`environment/legacy/AGENTS.md`**.
