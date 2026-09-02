// CSD_RATIONALE_BEGIN
// H84 GSM-2B diagnostic probe.
//
// Purpose:
// - H80 allowed route labels and prose around visible candidate spans; H83 then
//   showed that direct outputs contained partial span evidence, but not a
//   broad clean selector-ready pool.
// - H84 changes the candidate-generation contract: emit bare machine-readable
//   arithmetic expressions in visible constrained spans, with labels outside
//   spans only.
//
// Fairness / scope:
// - Local Qwen3.5-2B mechanism probe only; not a publishable result by itself.
// - No grammar, scorer, split, baseline model, or held-out path change.
// - No billed provider credentials are needed or used.
// CSD_RATIONALE_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var spanCount: nat := 0;
var maxCandidateSpans: nat := 6;
var closeBudgetPerSpan: nat := 40;

helpers.AppendTaskGuidance(
  lm,
  "Write exactly six candidate lines and no other explanation. Each line must have a label outside the span and one bare arithmetic expression inside the span: A: <<expr>>, B: <<expr>>, C: <<expr>>, D: <<expr>>, E: <<expr>>, F: <<expr>>. Inside each << >> span use only variable names from the problem, numbers, parentheses, and arithmetic operators + - * / // %. Do not put words, units, route descriptions, LaTeX, equals signs, or repeated text inside spans. After the six candidate lines, write Final: <<best_expr>> using one of the candidate expressions or a simplified equivalent."
);

while steps < maxSteps && spanCount < maxCandidateSpans
  invariant 0 <= steps <= maxSteps
  invariant 0 <= spanCount <= maxCandidateSpans
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps, maxCandidateSpans - spanCount
{
  var remaining: nat := maxSteps - steps;
  var chunkSize: nat := remaining;
  if chunkSize > 48 {
    chunkSize := 48;
  }
  if chunkSize == 0 {
    break;
  }

  var genOut, stoppedOnOpen, stoppedOnEos, stepsUsed :=
    helpers.UnconstrainedChunk(lm, prompt, generated, chunkSize, "<<", eosToken);
  generated := genOut;
  steps := steps + stepsUsed;

  if stoppedOnOpen && steps < maxSteps {
    generated, insideConstrainedOut, currentConstrainedOut :=
      helpers.EnterObservedConstrainedSpan(lm, generated);

    var closeBudget: nat := maxSteps - steps;
    if closeBudget > closeBudgetPerSpan {
      closeBudget := closeBudgetPerSpan;
    }

    var cg, ci, cc := helpers.CloseSpanWithinBudget(
      lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
    );
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + closeBudget;
    spanCount := spanCount + 1;
  } else if stoppedOnEos {
    break;
  }
}

// Force one final parser-checked answer if the model never opened a candidate
// span naturally.
if !insideConstrainedOut && steps < maxSteps && spanCount == 0 {
  generated, insideConstrainedOut, currentConstrainedOut :=
    helpers.OpenConstrainedSpan(lm, generated);
  steps := steps + 1;
}

if insideConstrainedOut && steps < maxSteps {
  var finalBudget: nat := maxSteps - steps;
  if finalBudget > closeBudgetPerSpan {
    finalBudget := closeBudgetPerSpan;
  }
  var fg, fi, fc := helpers.CloseSpanWithinBudget(
    lm, parser, prompt, generated, currentConstrainedOut, eosToken, finalBudget
  );
  generated := fg;
  insideConstrainedOut := fi;
  currentConstrainedOut := fc;
  steps := steps + finalBudget;
}

assert steps <= maxSteps;
assert |generated| <= |generatedPrefix| + steps;

cost := steps;
