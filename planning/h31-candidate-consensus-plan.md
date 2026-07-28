# H31 candidate consensus selector plan

**Date:** 2026-06-29

**Task:** Turn the H27/H28 no-gold agreement selector into a small tested framework module before any new GSM model run.

## Inputs

```
candidate records
  ├─ group id: problem/example id
  ├─ expression: candidate answer text
  ├─ equivalence key: no-gold normalized/symbolic key supplied by caller
  ├─ source: exact candidate source or attempt id
  ├─ source family: broader source family, if known
  └─ quality score: non-gold score such as H21 parse/grounding/simplicity
```

No expected answers, correctness labels, evaluator results, or benchmark-specific gold data may enter the selector.

## Outputs

```
selection result per group
  ├─ selected candidate
  ├─ chosen cluster key
  ├─ cluster agreement score
  ├─ source/source-family counts
  └─ audit fields explaining the choice
```

## Algorithm

1. Group candidates by example id.
2. Within each example, cluster candidates by caller-supplied equivalence key.
3. Score each cluster using no-gold signals:
   - total candidate support;
   - distinct exact sources;
   - distinct source families;
   - best candidate quality score as a tie-breaker.
4. Pick the best cluster.
5. Inside the chosen cluster, pick the candidate with the best quality score.
6. Return an audit-friendly result so later diagnostics can explain each selection.

## Verification

1. Write tests first and confirm they fail because the selector does not exist yet.
2. Implement the smallest low-dependency selector module.
3. Re-run the focused tests until green.
4. Run a sibling search for existing selector/consensus code to avoid duplicate conflicting paths.

## Non-goals for H31

- No model calls.
- No billed API calls.
- No GSM-specific expected-answer logic.
- No benchmark win claim.
- No held-out result claim.

