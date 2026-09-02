// CSD_RATIONALE_BEGIN
// H80 GSM-2B diagnostic probe.
//
// Purpose:
// - H64 proved the structured-candidate artifact path works, but many intended
//   candidate expressions appeared as prose or LaTeX-like text.
// - H80 changes the candidate exposure mechanism: route candidates should be
//   visible constrained spans, so the post-run artifact can recover arithmetic
//   expressions from `<<...>>` spans without using expected answers.
//
// Fairness / scope:
// - Local Qwen3.5-2B mechanism probe only; not a publishable result by itself.
// - No grammar, scorer, split, baseline prompt, or model change.
// - No billed provider credentials are needed or used.
// CSD_RATIONALE_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var spanCount: nat := 0;
var maxCandidateSpans: nat := 6;
var closeBudgetPerSpan: nat := 48;

helpers.AppendTaskGuidance(
  lm,
  "Create up to six independent arithmetic candidate expressions. Use route labels A through F in plain text, and put each candidate expression itself inside its own visible constrained span like Candidate A: <<expr>>. The text outside spans can explain the route briefly, but inside each << >> span use only variables, numbers, parentheses, and arithmetic operators. No words, units, LaTeX, equals signs with prose, or repeated junk inside spans. After the candidate spans, choose the simplest candidate supported by the routes and put the final chosen expression in one last << >> span."
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
  if chunkSize > 64 {
    chunkSize := 64;
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

// If no candidate span was opened naturally, force one final constrained span so
// the evaluator still receives a parser-checkable answer.
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
