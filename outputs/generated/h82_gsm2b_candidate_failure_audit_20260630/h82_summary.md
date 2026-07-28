# H82 GSM-2B candidate failure audit

Created: 2026-06-30T06:17:14.109447+00:00

This was CPU-only: **0** model calls, **0** GPU calls, **0** billed API calls, and no score artifact edits.

## Result

H82 confirms a candidate-pool quality/generation blocker, not another label-format blocker.

- H64 direct accuracy: **0.04081632653061224** with syntax **0.9387755102040817** on **49** examples.
- H64 extracted **218** candidate records across **46** groups, but only **1** `candidate_line` records look machine-parseable by a conservative text heuristic. Most candidate lines are prose or LaTeX-like descriptions.
- H64 `final_actual` candidates cover **46** groups, but only **2/49** direct final answers were correct.
- H80 direct accuracy: **0.02040816326530612** with syntax **0.5102040816326531** on **49** examples.
- H80 structured candidate count: **0** across **0** groups.

## Interpretation

The next GSM-2B hypothesis should change candidate generation so each candidate is emitted as a bare, machine-readable arithmetic expression with variable names preserved. Do **not** rerun H64's prose route labels or H80's visible-span body. A selector cannot rescue this branch until the candidate pool contains clean expressions.

Credential key-name scan hits: **0**.

Artifacts audited:

- `outputs/generated/h64_gsm2b_named_route_structured_candidates_20260629_t0/results/direct_eval_success.json`
- `outputs/generated/h64_gsm2b_named_route_structured_candidates_20260629_t0/results/structured_candidates.json`
- `outputs/generated/h80_gsm2b_visible_span_candidates_20260629_t0/results/direct_eval_success.json`
- `outputs/generated/h80_gsm2b_visible_span_candidates_20260629_t0/results/structured_span_candidates.json`
