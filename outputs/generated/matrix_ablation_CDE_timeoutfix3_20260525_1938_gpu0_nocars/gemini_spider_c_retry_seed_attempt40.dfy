// CSD_RATIONALE_BEGIN
// The best-performing strategies (Attempts 29, 39) have stabilized at 30% accuracy, limited by `syntax_valid_semantic_mismatch` errors. These strategies use `AdaptiveConstrainedStepWithPenalties`, which strongly boosts schema tokens to ensure syntactic correctness. While effective for syntax, this may be counterproductive for semantics. The hypothesis is that the aggressive schema boost is forcing the model to choose syntactically valid but semantically incorrect tokens (e.g., wrong table name), overriding the model's understanding of the question.
//
// This attempt tests this hypothesis by removing the schema boost entirely. We will replace `AdaptiveConstrainedStepWithPenalties` with the simpler `ConstrainedStep`. This new strategy retains the successful structure of immediately entering a constrained span to guarantee the output format and using the parser for hard grammatical constraints. However, by using `ConstrainedStep`, the choice among valid next tokens will be guided solely by the model's own logits, which are conditioned on the semantic content of the user's question. This gives the model's semantic reasoning more influence, which should help reduce `syntax_valid_semantic_mismatch` errors and improve accuracy, while the parser continues to guarantee a high syntax rate.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: `insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)`
//   - Unconstrained branch (`!insideConstrainedOut`): `OpenConstrainedSpan` is called, setting `insideConstrainedOut := true` and `currentConstrainedOut := []`. The method precondition `parser.IsValidPrefix([])` ensures this is valid.
//   - Completion branch (`IsCompletePrefix`): `CloseConstrainedSpan` sets `insideConstrainedOut := false`, making the invariant's implication vacuous for the next iteration.
//   - Generation branch: `ConstrainedStep` returns a token that is guaranteed by its contract to be valid if it's not `eosToken`. `AppendConstrainedToken` is then called with this valid token, preserving the prefix's validity.
// progress: `|generated| <= |generatedPrefix| + steps`
//   - Each path through the loop's `if/else` structure increments `steps` by exactly 1.
//   - The helpers `OpenConstrainedSpan`, `CloseConstrainedSpan`, and `ConstrainedStep` (via `AppendConstrainedToken`) append at most 1 token to `generated`.
//   - In all cases, the change in `|generated|` is less than or equal to the change in `steps` (which is 1), so the invariant `|generated| <= |generatedPrefix| + steps` is preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Generate a single SQL query that correctly answers the user's question based on the provided schema.");

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
    var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);
    generated := openedGenerated;
    insideConstrainedOut := openedInside;
    currentConstrainedOut := openedCurrent;
    steps := steps + 1;
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
    break;
  } else {
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.ConstrainedStep(
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
