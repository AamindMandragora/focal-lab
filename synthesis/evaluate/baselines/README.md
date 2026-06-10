# Baseline adapters

Each fixed strategy has a dedicated adapter module under this package. Adapters
invoke **patched legacy repos** (`legacy/CRANE`, `legacy/itergen`, `legacy/cars`)
or vendored SynCode (GCD / RS) through `run_legacy_fixed_strategy.py` helpers,
then emit normalized JSON under `outputs/baselines/<model>/<benchmark>/<strategy>/`.

## Layout

| Module | Strategy | Legacy source |
|--------|----------|---------------|
| `unconstrained.py` | unconstrained | `legacy/CRANE` (`main.py`, original mode) or Spider vLLM |
| `gcd.py` | gcd | vendored SynCode (`grammar_strict`) |
| `crane.py` | crane | `legacy/CRANE` (`main.py`, adaptive mode) |
| `itergen.py` | itergen | `legacy/itergen` |
| `cars.py` | cars | `legacy/cars` — see **[CARS_SETUP.md](CARS_SETUP.md)** for clone/patch/run and speed notes |
| `rs.py` | rs | vendored SynCode (`mode=original`, temp 1) |
| `smiles.py` | all (SMILES dataset) | pooled native protocol (`benchmarks/smiles/pooled_baseline.py`) |

Dispatch entrypoint: `registry.run_baseline_strategy(args)`.

Patches applied after clone: `environment/legacy_patches/{CRANE,itergen,cars}/*.patch`
via `environment/clone_legacy_csds.sh`.
