// CSD_RATIONALE_BEGIN
// The previous attempt (35) successfully used `UnconstrainedChunk` to solve the delimiter-triggering problem that plagued `UnconstrainedStep` strategies, achieving a high syntax rate (82.4%) on non-failing examples. However, it re-introduced a critical performance issue, causing 3 out of 17 examples to time out. This indicates that the chosen chunk budget of 32 tokens was too large, leading to excessively long LM generation calls in the unconstrained phase.
//
// The primary goal of this revision is to eliminate the timeouts while retaining the robust delimiter detection of `UnconstrainedChunk`. The hypothesis is that a significantly smaller chunk budget will prevent timeouts by breaking up the unconstrained generation into shorter, faster calls. This strategy therefore retains the exact structure of the previous one but makes a single, critical parameter change: the `chunkBudget` for `UnconstrainedChunk` is reduced from 32 to 8. This change is a direct and targeted fix for the `too_slow` failure mode, aiming to find a better balance between robust triggering and performance. The sophisticated and syntactically successful hybrid logic for the constrained generation phase remains unchanged.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: `insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)`
//   - Unconstrained branch (`!insideConstrainedOut`): The strategy uses `UnconstrainedChunk`. It transitions to constrained mode only if `stoppedOnOpenSpan` is true. At that point, `currentConstrainedOut` is set to `[]`, which is a valid prefix by precondition, so the invariant holds.
//   - Constrained branches (`insideConstrainedOut`):
//     - `IsCompletePrefix`: `CloseConstrainedSpan` makes the invariant vacuously true by setting `insideConstrainedOut` to false.
//     - `|currentConstrainedOut| >= rollbackLimit`: `RollbackConstrainedSuffix`'s postcondition guarantees the new `currentConstrainedOut` is a valid prefix.
//     - Narrow path (`validCount <= narrowThreshold`): `AdaptiveConstrainedStepWithPenalties` returns a valid next token, and `AppendConstrainedToken` preserves validity.
//     - Wide path: `ConstrainedSymbolInGenerated`'s postcondition guarantees the new `currentConstrainedOut` is a valid prefix.
//
// progress: `|generated| <= |generatedPrefix| + steps`
//   - Unconstrained branch: `UnconstrainedChunk` updates `steps` by `stepsUsed`. Its postcondition ensures the growth in `|generated|` is bounded by `stepsUsed`, preserving the invariant.
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
    var remainingBudget := maxSteps - steps;
    var chunkBudget: nat := if remainingBudget > 8 then 8 else remainingBudget; // Reduced from 32
    var chunkedG, stoppedOpen, stoppedEos, stepsUsed := helpers.UnconstrainedChunk(
        lm, prompt, generated, chunkBudget, "<<", eosToken
    );
    generated := chunkedG;
    steps := steps + stepsUsed;
    if stoppedEos {
        break;
    } else if stoppedOpen {
        insideConstrainedOut := true;
        currentConstrainedOut := [];
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
