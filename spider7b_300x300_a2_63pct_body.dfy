// CSD_RATIONALE_BEGIN
// Attempt 2 / attempt 8 / 9 / 14 / 16 strategy — confirmed 63%/100% training accuracy
// five times on the 300x300 seed334 train split.
//
// Core design: force OpenConstrainedSpan immediately, then use ConfidenceGatedStep
// with constrainedPrompt = prompt + ["<<"] so the model sees the delimiter context.
// Guidance prevents "SQL:" prefix and steers toward simplest correct query.
// CSD_RATIONALE_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

helpers.AppendTaskGuidance(lm, "Output only a valid SQL query. Start directly with SELECT (or WITH). Do not write 'SQL:', 'Answer:', or any prefix. Use only tables and columns mentioned in the schema. Write the simplest correct query.");

if steps < maxSteps && !insideConstrainedOut {
  var openGenerated, openInside, openCurrent := helpers.OpenConstrainedSpan(lm, generated);
  generated := openGenerated;
  insideConstrainedOut := openInside;
  currentConstrainedOut := openCurrent;
  steps := steps + 1;
}

while steps < maxSteps && insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  if parser.IsCompletePrefix(currentConstrainedOut) {
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
