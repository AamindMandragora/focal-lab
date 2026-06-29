    generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve step by step. Write each intermediate calculation and the final answer inside << >> delimiters.");

var steps: nat := 0;
var spanSteps: nat := 0;

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
    var chunkBudget := maxSteps - steps;
    if chunkBudget > 15 {
      chunkBudget := 15;
    }
    var chunkGenerated, stoppedOnOpenSpan, stoppedOnEos, stepsUsed := helpers.UnconstrainedChunk(
      lm, prompt, generated, chunkBudget, "<<", eosToken
    );
    steps := steps + stepsUsed;
    generated := chunkGenerated;
    if stoppedOnEos {
      break;
    } else if stoppedOnOpenSpan {
      var enteredGenerated, enteredInside, enteredCurrent := helpers.EnterObservedConstrainedSpan(
        lm, generated
      );
      generated := enteredGenerated;
      insideConstrainedOut := enteredInside;
      currentConstrainedOut := enteredCurrent;
      spanSteps := 0;
    }
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
    spanSteps := 0;
  } else {
    // Check per-span budget first (before taking a step)
    if spanSteps >= 40 {
      // Force rollback and exit — count as one step to ensure progress
      var rolledGenerated, rolledCurrent := helpers.RollbackConstrainedSuffix(
        parser, generated, currentConstrainedOut
      );
      generated := rolledGenerated;
      currentConstrainedOut := rolledCurrent;
      steps := steps + 1;
      if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
        var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
          lm, parser, generated, currentConstrainedOut
        );
        generated := closedGenerated;
        insideConstrainedOut := closedInside;
        currentConstrainedOut := closedCurrent;
        steps := steps + 1;
      } else {
        insideConstrainedOut := false;
        currentConstrainedOut := [];
      }
      spanSteps := 0;
    } else {
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.AdaptiveConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, 12, eosToken
      );
      steps := steps + 1;
      spanSteps := spanSteps + 1;
      if next == eosToken {
        var rolledGenerated, rolledCurrent := helpers.RollbackConstrainedSuffix(
          parser, generated, currentConstrainedOut
        );
        generated := rolledGenerated;
        currentConstrainedOut := rolledCurrent;
        if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
          var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
            lm, parser, generated, currentConstrainedOut
          );
          generated := closedGenerated;
          insideConstrainedOut := closedInside;
          currentConstrainedOut := closedCurrent;
          steps := steps + 1;
        } else {
          insideConstrainedOut := false;
          currentConstrainedOut := [];
        }
        spanSteps := 0;
      } else {
        var valid := helpers.IsTokenValidNext(parser, currentConstrainedOut, next);
        if valid {
          var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
            lm, parser, generated, currentConstrainedOut, next
          );
          generated := appendedGenerated;
          insideConstrainedOut := appendedInside;
          currentConstrainedOut := appendedCurrent;
          if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
            var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
              lm, parser, generated, currentConstrainedOut
            );
            generated := closedGenerated;
            insideConstrainedOut := closedInside;
            currentConstrainedOut := closedCurrent;
            steps := steps + 1;
            spanSteps := 0;
          }
        }
      }
    }
  }
}

cost := steps;
