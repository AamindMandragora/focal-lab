# H88 H86 SMILES failure-mode audit
Date: 2026-06-30
Scope: CPU-only read of active H86 log/run artifacts. No model, GPU, or paid API call.
## Result
- Latest H86 attempt marker: 16 / 40.
- Best/recent anchors: [{'attempt': 8, 'accuracy_pct': 34.0, 'syntax_pct': 41.0}, {'attempt': 8, 'accuracy_pct': 34.0, 'syntax_pct': 41.0}, {'attempt': 8, 'accuracy_pct': 34.0, 'syntax_pct': 41.0}].
- Pattern counts: `{"context_overflow": 33, "entered_constrained_mode_too_early": 25, "long_invalid_concatenated_smiles": 32, "prompt_duplicate_or_exemplar": 18, "tiny_span_dominant": 9, "unique_valid": 22, "validity_or_syntax": 102}`.
- Generated CSD files inspected: 12.
- Top helper calls: `[]`.
- Classification: `['span/control-output-budget', 'long-invalid-concatenated-smiles', 'duplicate/diversity']`.
- Recommendation: Prioritize a general SMILES single-output/span-control helper or output-budget guard before another duplicate/canonicalization helper.

## Evidence snippets
### context_overflow
```text
// 1. The main failure mode (58/100 examples) is "runtime_or_generation_error" due to
//    context length overflow: the unconstrained preamble causes the model to generate
//    very long outputs before entering the constrained span, exceeding the 16384 token limit.
```
```text
//    handle it, but the preamble itself is costly.
// 3. The context overflow is caused by: (a) the long task guidance string being injected,
//    and (b) the unconstrained preamble generating many tokens.
```
```text
//
// Key insight: The 58 failures are ALL due to context overflow. The 42 that succeed have
// accuracy 40/41 = 97.6% when they produce valid output. So the strategy itself is correct
```
### entered_constrained_mode_too_early
```text
// Diagnosis: All three prior attempts produce "tiny spans" containing only "C"
// (or similar minimal SMILES), and "entered_constrained_mode_too_early".
// The model sees "isocyanates" as the prompt, immediately generates "<<" when
```
```text
// We need to force the model to generate a more complex SMILES that actually
// contains the isocyanate motif (N=C=O). The constrained span opens too early
// before the model has any context to know what kind of SMILES to generate.
```
```text
// model has context before entering the constrained SMILES span.
// This prevents "entered_constrained_mode_too_early" and gives the LM a chance
// to reason about what isocyanate SMILES to produce.
```
### tiny_span_dominant
```text
Strategy: // CSD_RATIONALE_BEGIN
// Diagnosis: All three prior attempts produce "tiny spans" containing only "C"
// (or similar minimal SMILES), and "entered_constrained_mode_too_early".
```
```text
Strategy: // CSD_RATIONALE_BEGIN
// The core problem is "entered_constrained_mode_too_early" and "tiny_span_dominant":
// the model opens the constrained span immediately and generates only 1-2 tokens
```
```text
// This produces a chemically valid but semantically wrong output (methane, not
// an isocyanate). The tiny_span_dominant: 100% diagnostic confirms this.
//
```
### long_invalid_concatenated_smiles
```text
  [EVAL]   Generated 400 tokens in 46.47s
[10:10:18] SMILES Parse Error: unclosed ring for input: 'C1=CC=C(C=C1)N=C=ON=C=O1.CCOC(CN=C=O)C(=O)N=C=O.CC1=CC=C(C=C1)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)'
[10:10:18] SMILES Parse Error: unclosed ring for input: 'C1=CC=C(C=C1)N=C=ON=C=O1.CCOC(CN=C=O)C(=O)N=C=O.CC1=CC=C(C=C1)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(
```
```text
[10:10:18] SMILES Parse Error: unclosed ring for input: 'C1=CC=C(C=C1)N=C=ON=C=O1.CCOC(CN=C=O)C(=O)N=C=O.CC1=CC=C(C=C1)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)'
[10:10:18] SMILES Parse Error: unclosed ring for input: 'C1=CC=C(C=C1)N=C=ON=C=O1.CCOC(CN=C=O)C(=O)N=C=O.CC1=CC=C(C=C1)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(CN=C=O)C(=O)N=C=O.CC(
```
```text
  [EVAL]   Generated 400 tokens in 46.19s
[10:11:04] SMILES Parse Error: syntax error while parsing: C1=CC=C(C=C1)N=C=O.0=1C(=O)N2C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N1C(=O)N(C)C(=O)N
[10:11:04] SMILES Parse Error: check for mistakes around position 20:
```
### prompt_duplicate_or_exemplar
```text
Helper policy: helper mask warm-up (0/3 evaluated attempts for bandit)
Generating initial strategy for: Generate one new, valid, non-exemplar SMILES molecule for the isocyanates class. The answer contract is a single SMILES string and nothing else. Use the hidden parser-guided constrained chunk for that SMILES token sequence and avoid copying prompt exemplars.

```
```text
// 1. Emit task guidance to steer the model toward generating a novel isocyanate
//    SMILES (not copying exemplars). Isocyanates contain the N=C=O motif.
// 2. Immediately open a hidden constrained span (no visible "<<" / ">>" delimiters
```
```text
      } else {
        // Wider state: use repetition penalty to avoid copying exemplars
        next := helpers.SafeRepetitionPenaltyStep(
```

## Safety
- model_calls: 0
- gpu_calls: 0
- billed_api_calls: 0
- scorer/dataset/baseline edits: 0
