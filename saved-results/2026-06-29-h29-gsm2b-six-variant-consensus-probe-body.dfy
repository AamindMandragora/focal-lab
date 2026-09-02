// CSD_RATIONALE_BEGIN
// H29 GSM-2B diagnostic mechanism probe.
//
// Purpose:
// - Test the pre-registered H29 idea from docs/experiments/metadecode-fast-iteration-log.md.
// - H27/H28 showed that within-output candidate agreement is load-bearing for the GSM-2B
//   diagnostic selector; H24 only asked for three candidate equations and relied on the model's
//   weak self-selected answer.
// - Ask the local 2B model to write six independent compact candidate equations and a plain-text
//   consensus line. The post-run diagnostic will ignore the model's self-choice and evaluate
//   candidates with a non-gold agreement-first selector.
//
// Fairness / scope:
// - This is a $0 local mechanism probe, not a publishable synthesis result.
// - It does not edit the grammar, grader, dataset split, baseline prompts, or model.
// - It intentionally uses explicit candidate-variant guidance, so a good score here would only
//   motivate a later fair framework/helper change that the synthesizer can discover from neutral
//   inputs.
// CSD_RATIONALE_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

helpers.AppendTaskGuidance(
  lm,
  "Reason in plain text first. Before the final answer, write exactly six compact ASCII equation lines: Candidate 1: <expr>, Candidate 2: <expr>, Candidate 3: <expr>, Candidate 4: <expr>, Candidate 5: <expr>, Candidate 6: <expr>. Make the six candidates independent: derive them in different simple ways when possible, and do not copy a candidate unless another derivation gives the same expression. Each candidate expression must use only variables and arithmetic operators from the problem; no words, no units, no LaTeX, no explanation on those lines. Then write one line Consensus: <expr>, copying an expression that is supported by at least two candidates if any two agree; otherwise choose the grounded, simplest candidate with no repeated junk or huge numeric constants. Finally open exactly one << >> span and copy the Consensus expression exactly into the span. Do not put intermediate work inside << >>."
);

// Reserve a bounded budget for the final constrained span. The model gets most of
// the budget for plain-text candidate generation; the final span cannot grow into
// H1-style repeated algebra.
var finalReserve: nat := 64;
if finalReserve > maxSteps {
  finalReserve := maxSteps / 2;
}

var freeCap: nat := maxSteps - finalReserve;

// Phase 1: free reasoning and candidate generation, stopping early if the model
// starts the final span or EOS.
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

  if stoppedOnOpen {
    generated, insideConstrainedOut, currentConstrainedOut :=
      helpers.EnterObservedConstrainedSpan(lm, generated);
  } else if stoppedOnEos {
    break;
  }
}

// Phase 2: if the model did not open the final span itself, force one bounded
// final-answer span after the candidate list and Consensus line.
if !insideConstrainedOut && steps < maxSteps {
  generated, insideConstrainedOut, currentConstrainedOut :=
    helpers.OpenConstrainedSpan(lm, generated);
  steps := steps + 1;
}

// Phase 3: close the final span within the reserved budget.
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
