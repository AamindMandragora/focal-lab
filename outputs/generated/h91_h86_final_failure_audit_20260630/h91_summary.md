# H91 final H86 failure audit

Date: 2026-06-30

## Inputs

- Report: `outputs/generated/smiles_qwen35_9b_isocyanates_uv_qwen35_0627/smiles_qwen35_9b_isocyanates_uv_qwen35_0627_20260630_083748_22e3d6/results/failure_report.json`
- Log: `/tmp/csd_h86_logs/h86_smiles_qwen35_9b_isocyanates_train100_20260630.log`

## Outputs

- `h91_summary.json`
- This `h91_summary.md`

## Algorithm

1. Load the completed H86 failure report.
2. Compare attempt-level accuracy and syntax.
3. Inspect sample-level SMILES fields for the best-accuracy attempt, highest-syntax attempt, and final attempt.
4. Count known failure-pattern strings in the run log.
5. Choose the next fair SMILES lever from the observed failure shape.

## Result

- Total attempts: **40**
- Best accuracy attempt: **20**, accuracy **0.37**, syntax **0.41**
- Highest syntax attempt: **1**, accuracy **0.00**, syntax **1.00**
- Final metric attempt: **40**, accuracy **0.10**, syntax **0.25**

## Sample-level finding

Highest-syntax attempt **1** had:

- Syntax-valid samples: **100/100**
- Class-membership samples: **0/100**
- Unique-valid candidates: **0/100**
- Prompt exemplars: **0/100**

Best-accuracy attempt **20** had:

- RDKit-valid samples: **41/100**
- Unique-valid candidates: **41/100**
- Mean output length: **490.2** characters
- Max output length: **558** characters

## Log-pattern counts

```json
{
  "context_overflow": 52,
  "duplicate_or_exemplar": 225,
  "entered_constrained_mode_too_early": 38,
  "long_invalid_concatenated_smiles": 10,
  "managed_span_helper": 2,
  "prefix_appears_helper": 0,
  "tiny_span_dominant": 43
}
```

## Recommendation

Next lever: **new_or_repaired_general_class_membership_and_candidate_selection_helper**

Reason: The high-syntax attempt produced many syntactically valid molecules, but only a minority were unique valid in-class candidates. This points beyond span mechanics alone.

## Safety

This audit made **0** model calls, **0** GPU calls, **0** billed API calls, and **0** score-artifact edits.
