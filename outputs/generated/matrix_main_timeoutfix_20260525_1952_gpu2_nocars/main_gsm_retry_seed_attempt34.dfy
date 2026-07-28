// CSD_RATIONALE_BEGIN
// The previous attempt (33) introduced a `markerToken` to proactively trigger constrained generation, but evaluation data showed this was ineffective, with most failures still not using the constrained logic (`used_constrained: no`). This led to a significant regression in accuracy (12.0%) compared to the best-performing strategy (Attempt 16, 25.0% accuracy). The marker-based heuristic proved too brittle and less reliable than simply detecting the `<<` delimiter.
//
// This revision reverts to the proven strategy of Attempt 16. This strategy combines a robust, token-by-token `UnconstrainedStep` loop to reliably enter constrained mode, with a sophisticated hybrid logic for generation inside the span. This internal logic uses `AdaptiveConstrainedStepWithPenalties` for precision in narrow states, `ConstrainedSymbolInGenerated` for efficiency in wider states, and `RollbackConstrainedSuffix` for safety against getting stuck. By returning to the most successful configuration, this strategy aims to re-establish the higher performance baseline and fix the triggering failures that plagued recent attempts.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: `insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)`
//   - Unconstrained branch (`!insideConstrainedOut`): The strategy uses `UnconstrainedStep` and transitions to constrained mode only if the generated token is `<<`. At that point, `currentConstrainedOut` is set to `[]`, which is a valid prefix by precondition.
//   - Constrained branches (`insideConstrainedOut`):
//     - `IsCompletePrefix`: `CloseConstrainedSpan` makes the invariant vacuously true by setting `insideConstrainedOut` to false.
//     - `|currentConstrainedOut| >= rollbackLimit`: `RollbackConstrainedSuffix`'s postcondition guarantees the new `currentConstrainedOut` is a valid prefix.
//     - Narrow path (`validCount <= narrowThreshold`): `AdaptiveConstrainedStepWithPenalties` returns a valid next token, and `AppendConstrainedToken` preserves validity.
//     - Wide path: `ConstrainedSymbolInGenerated`'s postcondition guarantees the new `currentConstrainedOut` is a valid prefix.
//
// progress: `|generated| <= |generatedPrefix| + steps`
//   - Unconstrained branch: `UnconstrainedStep` consumes 1 step, and `|generated|` grows by at most 1, preserving the invariant.
//   - `ConstrainedSymbolInGenerated`: Updates `steps` by `stepsUsed`, and the growth in `|generated|` is bounded by `stepsUsed`, preserving the invariant.
//   - `CloseConstrainedSpan`, `AdaptiveConstrainedStepWithPenalties`, and `RollbackConstrainedSuffix`: Each increments `steps` by 1 and changes `|generated|` by at most 1 (or shrinks it in the rollback case), preserving the invariant.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var narrowThreshold: nat := 10;
var rollbackLimit: nat := 30;
var penaltyTokens: seq<Token> := ["/"];
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
  } else { // insideConstrainedOut
    if parser.IsCompletePrefix(currentConstrainedOut) {
      var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
        lm, parser, generated, currentConstrainedOut
      );
      generated := closedGenerated;
      insideConstrainedOut := closedInside;
      currentConstrainedOut := closedCurrent;
      steps := steps + 1;
    } else if |currentConstrainedOut| >= rollbackLimit {
      var rolledGenerated, rolledCurrent := helpers.RollbackConstrainedSuffix(
        parser, generated, currentConstrainedOut
      );
      generated := rolledGenerated;
      insideConstrainedOut := true;
      currentConstrainedOut := rolledCurrent;
      steps := steps + 1;
    } else {
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
      if validCount <= narrowThreshold {
        var next := helpers.AdaptiveConstrainedStepWithPenalties(
          lm, parser, constrainedPrompt, currentConstrainedOut,
          validTokenGroups, 4.0,
          penaltyTokens, 4.0,
          (narrowThreshold as int), eosToken
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
      } else {
        var remaining: nat := maxSteps - steps;
        var symbolBudget: nat := if stepTokenBudget == 0 || stepTokenBudget > remaining then remaining else stepTokenBudget;
        var symbolGenerated, symbolOut, hitEos, stepsUsed := helpers.ConstrainedSymbolInGenerated(
          lm, parser, constrainedPrompt, generated, currentConstrainedOut, symbolBudget, eosToken
        );
        generated := symbolGenerated;
        currentConstrainedOut := symbolOut;
        steps := steps + stepsUsed;
        if hitEos {
          break;
        }
      }
    }
  }
}

cost := steps;
