// CSD_RATIONALE_BEGIN
// H10 GSM-2B diagnostic mechanism probe.
//
// Purpose:
// - Test the pre-registered H10 idea from docs/experiments/metadecode-fast-iteration-log.md:
//   GSM-2B may need a structured equation-line then copy scaffold, not more prose guidance.
// - Keep H4's free-text reasoning plus one bounded final constrained span.
// - Change only the guidance text: require one ASCII "Equation:" line before the final span,
//   then ask the final span to copy that expression exactly.
//
// Fairness / scope:
// - This is a $0 local mechanism probe, not a publishable synthesis result.
// - It does not edit the grammar, grader, dataset split, span-close policy, or baseline prompts.
// - It intentionally uses explicit equation/copy guidance, so a good score here would only motivate
//   a later fair framework/helper change that the synthesizer can discover from neutral inputs.
// CSD_RATIONALE_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

helpers.AppendTaskGuidance(
  lm,
  "Reason in plain text first. Before the final answer, write exactly one line in this format: Equation: <one compact ASCII symbolic expression>. The expression must use only variables and arithmetic operators from the problem; no words, no units, no explanation on that line. Then open exactly one << >> span and copy the expression after Equation: exactly into the span. Do not put intermediate work inside << >>."
);

// Reserve a bounded budget for the final constrained span. The model gets most of
// the budget for plain-text reasoning; the final span cannot grow into H1-style
// repeated algebra.
var finalReserve: nat := 64;
if finalReserve > maxSteps {
  finalReserve := maxSteps / 2;
}

var freeCap: nat := maxSteps - finalReserve;

// Phase 1: free reasoning, stopping early if the model starts the final span or EOS.
while steps < freeCap && !insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant steps <= freeCap
  invariant freeCap <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases freeCap - steps
{
  var remaining: nat := freeCap - steps;
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

  if stoppedOnOpen {
    generated, insideConstrainedOut, currentConstrainedOut :=
      helpers.EnterObservedConstrainedSpan(lm, generated);
  } else if stoppedOnEos {
    break;
  }
}

// Phase 2: if the model did not open the final span itself, force one bounded
// final-answer span after the free reasoning.
if !insideConstrainedOut && steps < maxSteps {
  generated, insideConstrainedOut, currentConstrainedOut :=
    helpers.OpenConstrainedSpan(lm, generated);
  steps := steps + 1;
}

// Phase 3: close the final span within the reserved budget. If the span cannot
// reach a complete parse, the helper leaves it open; that makes H10 falsifiable by
// syntax and unclosed-span counts.
if insideConstrainedOut && steps < maxSteps {
  var closeBudget: nat := maxSteps - steps;
  if closeBudget > finalReserve {
    closeBudget := finalReserve;
  }
  var cg, ci, cc := helpers.CloseSpanWithinBudget(
    lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
  );
  generated := cg;
  insideConstrainedOut := ci;
  currentConstrainedOut := cc;
  steps := steps + closeBudget;
}

assert steps <= maxSteps;
assert |generated| <= |generatedPrefix| + steps;

cost := steps;
