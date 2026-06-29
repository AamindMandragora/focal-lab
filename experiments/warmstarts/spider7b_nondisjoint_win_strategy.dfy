// CSD_RATIONALE_BEGIN
// Analysis of failures:
// - Best attempt (54% acc, 84% syntax) used UnconstrainedStep until "<<" then AdaptiveConstrainedStep.
// - Main failure modes: syntax_valid_semantic_mismatch (14), no_valid_visible_span (6).
// - The model naturally emits "SQL: <<" after the prompt guidance, so the unconstrained phase works.
// - Inside the constrained span, we're over-constraining: AdaptiveConstrainedStep hard-masks every token.
//   This pushes the model onto grammar-valid but semantically wrong paths.
// - Fix: use ConfidenceGatedStep inside the constrained span. This keeps the model's own top token
//   when it's already grammar-valid, only falling back to hard masking when needed.
//   This should improve accuracy (less over-constraining) while maintaining syntax validity.
// - Also: the span token length averages ~17 tokens, which is very short for complex SQL.
//   The model is completing too early or being cut off. We should NOT close the span eagerly
//   on IsCompletePrefix if the query seems too short - but we can't easily check that.
//   Instead, use ConfidenceGatedStep to let the model continue more naturally.
// - Keep the same unconstrained phase (wait for "<<"), same closing logic.
// - Use SafeRepetitionPenaltyStep occasionally to avoid token repetition in long queries.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity invariant:
//   - Outside span (!insideConstrainedOut): implication is vacuously true.
//     When next == "<<", we set currentConstrainedOut := [] which satisfies
//     parser.IsValidPrefix([]) by precondition.
//   - CloseConstrainedSpan branch: sets insideConstrainedOut to false, making
//     the implication vacuous. Invariant preserved.
//   - Inside constrained span: ConfidenceGatedStep returns either EOS or a
//     parser-valid next token (by its contract: it returns the model's top token
//     if valid, else hard-masks to valid+EOS). AppendConstrainedToken appends
//     that token to currentConstrainedOut, preserving parser.IsValidPrefix.
//
// progress invariant (|generated| <= |generatedPrefix| + steps):
//   - Outside span: UnconstrainedStep costs +1 step, appends at most 1 token.
//     |generated| grows by at most 1 = steps growth. Invariant preserved.
//   - CloseConstrainedSpan: costs +1 step, appends ">>" (1 token).
//     |generated| grows by 1 = steps growth. Invariant preserved.
//   - Inside constrained span (ConfidenceGatedStep path): costs +1 step, returns
//     one token. AppendConstrainedToken appends at most 1 token. |generated|
//     grows by at most 1 = steps growth. Invariant preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Output exactly one SQL query in the format: SQL: <<YOUR QUERY>>. Use only tables and columns from the provided schema. No explanation, no markdown, no extra text. The query must be valid SQL.");

var steps: nat := 0;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  if !insideConstrainedOut {
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      generated := generated + [next];
      if next == "<<" {
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }
    }
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  } else {
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next, wasConstrained := helpers.ConfidenceGatedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut, eosToken
    );
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }
  }
}

cost := steps;