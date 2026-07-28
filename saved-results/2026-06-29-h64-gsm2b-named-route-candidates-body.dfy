// CSD_RATIONALE_BEGIN
// H64 GSM-2B diagnostic launch plan.
//
// Purpose:
// - H60 showed the H37/H40-H42 temperature-repeat branch is too weak: only 4/49
//   oracle coverage, far below the 12/49 train bar.
// - H57 showed the H31 worktree can materialize selector-ready structured
//   candidate artifacts with source ids, sample ids, and scorer metadata while
//   excluding gold correctness fields.
// - H64 tests the next *future* GPU-smoke shape: generate fresh candidate lines
//   whose independence is forced by named derivation routes, then let the H57
//   post-run artifact path expose those candidates for no-gold selection.
//
// Fairness / scope:
// - This is a local Qwen3.5-2B mechanism probe, not a publishable result by
//   itself.
// - It does not edit the grammar, grader, dataset split, span-close policy,
//   baseline prompts, or model.
// - It intentionally uses explicit candidate-route guidance, so a good score
//   would motivate a later fair framework/helper change discovered from neutral
//   inputs rather than being promoted directly.
// CSD_RATIONALE_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

helpers.AppendTaskGuidance(
  lm,
  "Reason in plain text first. Before the final answer, write exactly six compact ASCII equation lines with these parser-readable labels and route names: Candidate A: direct equation: <expr>, Candidate B: reverse check: <expr>, Candidate C: grouping: <expr>, Candidate D: unit operation: <expr>, Candidate E: minimal form: <expr>, Candidate F: sanity case: <expr>. Each route must derive the expression in a different way when possible; do not copy a previous candidate unless the route independently gives the same expression. Each candidate expression must use only variables and arithmetic operators from the problem; no words, no units, no LaTeX, no explanation after the expression on those candidate lines. Then write one line Consensus: <expr>, copying an expression supported by at least two routes if any two agree; otherwise choose the grounded, simplest candidate with no repeated junk or huge numeric constants. Finally open exactly one << >> span and copy the Consensus expression exactly into the span. Do not put intermediate work inside << >>."
);

// Reserve a bounded budget for the final constrained span. The model gets most
// of the budget for plain-text candidate generation; the final span cannot grow
// into H1-style repeated algebra.
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
