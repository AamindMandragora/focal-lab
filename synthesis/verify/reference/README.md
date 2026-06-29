# Reference CSD strategies (Dafny)

Verified **formal specifications** that mirror published strategy families (Unconstrained, GCD/SynCode, CRANE, IterGen, CARS) using only `CSDHelpers` from `library/VerifiedAgentSynthesis.dfy`.

## Two evaluation paths

| Path | Entry point | Uses these files? |
|------|-------------|-------------------|
| **Legacy baselines** | `python -m synthesis.evaluate.run_legacy_fixed_strategy` | No — runs `legacy/CRANE`, `legacy/itergen`, `legacy/cars`, or vendored SynCode |
| **Reference baselines** | `python -m synthesis.evaluate.run_reference_strategy` | **Yes** — compiles a chosen `*.dfy` here to Python and evaluates via `Evaluator` |

Each file defines its own module (`ReferenceGcdCSD`, `ReferenceCraneCSD`, etc.) for standalone `dafny verify`. `run_reference_strategy` rewrites the module name to `GeneratedCSD` before compilation.

For a concise index of every `LM`, `Parser`, and `CSDHelpers` member, see [`../library/README.md`](../library/README.md).

## Files

| File | Role |
|------|------|
| `unconstrained.dfy` | Pure `UnconstrainedStep` loop with no grammar enforcement. |
| `gcd.dfy` | Greedy constrained decoding (SynCode-style): hard mask inside `<<`…`>>` from the first constrained token. |
| `crane.dfy` | CRANE-style: unconstrained prefix; inside span, `GroupBoostedConstrainedStep` with empty groups. |
| `crane_faithful.dfy` | Variant CRANE reference body. |
| `itergen.dfy` | IterGen-style per-token soft-then-hard constrained steps inside the span. |
| `cars.dfy` | CARS-style adaptive rejection sampling with rollback and penalties. |

## Design distinction: GCD vs CRANE

GCD forces constrained mode from the first token inside the span. CRANE allows unconstrained reasoning before `<<` and uses adaptive switching. On Spider/SMILES, CRANE can emit natural-language reasoning before `<<SELECT ...>>` or `<<SMILES>>`.

## Verify only

From the repository root:

```bash
dafny verify synthesis/verify/reference/unconstrained.dfy \
               synthesis/verify/reference/gcd.dfy \
               synthesis/verify/reference/crane.dfy \
               synthesis/verify/reference/itergen.dfy \
               synthesis/verify/reference/cars.dfy
```

## See also

- **`../README.md`** — verify stage overview.
- **`run_reference_strategy.py`** — compile + evaluate a reference strategy on any benchmark.
