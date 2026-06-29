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

Optional unified patches applied after clone live under **`environment/legacy_patches/`**
(see that folder’s README). **Policy:** any manual change under **`legacy/*`** must be
mirrored there — see **`environment/legacy/AGENTS.md`**.

To verify patches apply cleanly, remove a tree and re-run **`bash environment/clone_legacy_csds.sh`**.
