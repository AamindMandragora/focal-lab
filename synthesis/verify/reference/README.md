# Reference CSD strategies (Dafny)

Verified **example** strategies that mirror baseline families (CRANE, IterGen, CARS) using only `CSDHelpers` from `library/VerifiedAgentSynthesis.dfy`.

Each file defines its own module (`crane`, `itergen`, `cars`) so they can be verified together without duplicate `GeneratedCSD` names. To use with the pipeline, copy `MyCSDStrategy` into `library/GeneratedCSD.dfy` under `module GeneratedCSD`.

For a concise index of every `LM`, `Parser`, and `CSDHelpers` member (and module-level helpers), see [`../library/README.md`](../library/README.md).

## Files

| File | Role |
|------|------|
| `crane.dfy` | Unconstrained prefix; inside `<<`…`>>`, `GroupBoostedConstrainedStep` with empty groups (hard mask only). |
| `itergen.dfy` | Same outer loop; inside span, `ConstrainedSymbol` with stable-prefix rebuild (chunk + longest valid prefix). |
| `cars.dfy` | Same outer loop; inside span, `AdaptiveConstrainedStep` with `validTokenGroups` and `stepTokenBudget` as narrow threshold (default 8 when 0). |

## Verify

From the repository root:

```bash
dafny verify synthesis/verify/reference/crane.dfy \
               synthesis/verify/reference/itergen.dfy \
               synthesis/verify/reference/cars.dfy
```

Or verify the whole directory in one invocation (same command with all three paths, or glob if your shell expands it).
