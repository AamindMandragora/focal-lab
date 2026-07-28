// CSD_RATIONALE_BEGIN
// Analysis of failures:
// 1. 9/49 examples have unterminated spans - the span opens near the end of budget
//    and runs out of steps before closing. The output tail shows things like
//    "<<n1 * price" or "<<t * currency" - the span opened but couldn't finish.
// 2. 37/49 examples have wrong answers - the model generates a plausible but
//    incorrect expression inside the span.
//
// Root causes:
// (A) Unterminated spans: The previous attempt spent too much budget on unconstrained
//     generation (75%), leaving too little for constrained close. When the model
//     emits "<<" late (near step 630/900), the remaining budget isn't enough.
//     Fix: Use CloseSpanWithinBudget with ALL remaining budget, not just 30 or 40.
//     Also reduce unconstrained phase to 60% of budget.
//
// (B) Wrong answers: The model's reasoning is often correct but the final expression
//     inside << >> doesn't match the correct formula. We need better guidance
//     to produce the right expression, and use the constrained generation to
//     force correct expression structure.
//
// Key architectural change:
// - Reduce phase1 unconstrained to 60% of budget
// - When inside constrained span, use ALL remaining budget for CloseSpanWithinBudget
//   (not just 30-40 steps)
// - This ensures spans that open late can still close
// - Keep the minInnerSteps=0 (don't enforce minimum before closing) to avoid
//   the "tiny span" problem - the parser will generate the right length
// - Use better guidance that emphasizes the final answer format
//
// The best result was attempt 11 (syntax 100%) which had the same architecture
// but without the minInnerSteps. The problem there was accuracy 2% only.
// Attempt 21 added minInnerSteps=6 and got accuracy 6.1% but broke syntax.
//
// The real accuracy problem is that the model isn't generating the right
// expression. We need to:
// 1. Ensure span closes (fix unterminated - use full remaining budget)
// 2. Generate a more complete expression (better guidance, ConstrainedGeneration
//    approach for the span content)
//
// Strategy:
// Phase 1: Unconstrained chunks watching for "<<" (60% of budget)
// Phase 2: Constrained steps with CloseSpanIfComplete 
// Phase 3: CloseSpanWithinBudget with ALL remaining budget
// Phase 4: If no span seen, more unconstrained (up to 80% total)
// Phase 5: Same constrained + close
// Phase 6: Force open + close with ALL remaining
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   Phase 1: UnconstrainedChunk stays outside span until stoppedOnOpen; 
//     EnterObservedConstrainedSpan sets currentConstrainedOut := [] which is valid.
//   Phase 2: CloseSpanIfComplete either closes (insideConstrainedOut false, implication vacuous)
//     or is a no-op. ConstrainedStep + AppendConstrainedToken preserves IsValidPrefix by
//     ConstrainedStep postcondition.
//   Phase 3: CloseSpanWithinBudget postcondition: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut).
//   Phase 4: Same as Phase 1.
//   Phase 5: Same as Phase 2.
//   Phase 6: Same as Phase 3.
//   Phase 7: OpenConstrainedSpan sets currentConstrainedOut := [] (valid).
//     CloseSpanWithinBudget postcondition preserves invariant.
//
// progress (|generated| <= |generatedPrefix| + steps):
//   Phase 1: UnconstrainedChunk: |generatedOut| <= |generated| + stepsUsed, steps += stepsUsed.
//   Phase 2: Each iteration: steps += 1, |generated| grows by at most 1 token.
//     So |generated| <= |generatedPrefix| + steps preserved.
//   Phase 3: CloseSpanWithinBudget: |generatedOut| <= |generated| + budget, steps += budget.
//   Phase 4: UnconstrainedStep: steps += 1, |generated| grows by at most 1.
//   Phase 5: Same as Phase 2.
//   Phase 6: CloseSpanWithinBudget: same as Phase 3.
//   Phase 7: OpenConstrainedSpan: +1 step, +1 token visible. 
//     CloseSpanWithinBudget: |generatedOut| <= |generated| + remainBudget, steps += remainBudget.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

if maxSteps == 0 {
  return;
}

var guidance: string := "Solve step by step using the symbolic variable names from the problem (like n1, n2, price, rate, etc.). At the very end, write exactly: The answer is <<EXPR>> where EXPR is an arithmetic expression using only those variable names, numbers, and operators +, -, *, /, (, ). No LaTeX, no braces {}, no $, no **. Keep the expression concise and correct. Example: The answer is <<(n1 + n2) * price / 60>>";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;
var hasCompletedSpan: bool := false;
var chunkSize: nat := 40;

// Use 60% of budget for unconstrained phase
var phase1Limit: nat := (maxSteps * 60) / 100;
if phase1Limit == 0 {
  phase1Limit := 1;
}

// Phase 1: Unconstrained generation in chunks watching for "<<" opening delimiter
while steps < phase1Limit && !insideConstrainedOut && !hasCompletedSpan
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases phase1Limit - steps
{
  var actualChunk := chunkSize;
  if steps + actualChunk > phase1Limit {
    actualChunk := phase1Limit - steps;
  }
  if actualChunk == 0 {
    break;
  }
  var cg, stoppedOnOpen, stoppedOnEos, stepsUsed := helpers.UnconstrainedChunk(lm, prompt, generated, actualChunk, "<<", eosToken);
  generated := cg;
  steps := steps + stepsUsed;
  if stoppedOnEos {
    break;
  }
  if stoppedOnOpen {
    var eg, ei, ec := helpers.EnterObservedConstrainedSpan(lm, generated);
    generated := eg;
    insideConstrainedOut := ei;
    currentConstrainedOut := ec;
  }
}

// Phase 2: If inside a constrained span, generate with ConstrainedStep
var innerStepLimit: nat := 60;
var innerSteps: nat := 0;

while insideConstrainedOut && !hasCompletedSpan && steps < maxSteps && innerSteps < innerStepLimit
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases (maxSteps - steps) + (innerStepLimit - innerSteps)
{
  var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
  if closed {
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + 1;
    hasCompletedSpan := true;
  } else {
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
    steps := steps + 1;
    innerSteps := innerSteps + 1;
    if next == eosToken {
      break;
    } else {
      var ag, ai, ac := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
      generated := ag;
      insideConstrainedOut := ai;
      currentConstrainedOut := ac;
    }
  }
}

// Phase 3: If span still open after inner loop, use CloseSpanWithinBudget with ALL remaining budget
if insideConstrainedOut && steps < maxSteps {
  var remaining := maxSteps - steps;
  var spanBudget: nat := remaining;
  // Cap at 200 to leave room for re-entry if needed
  if spanBudget > 200 {
    spanBudget := 200;
  }
  if spanBudget > 0 {
    var wg, wi, wc := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, spanBudget);
    generated := wg;
    insideConstrainedOut := wi;
    currentConstrainedOut := wc;
    steps := steps + spanBudget;
    if !insideConstrainedOut {
      hasCompletedSpan := true;
    }
  }
}

// Phase 4: If no span completed, continue free generation watching for "<<"
// Use up to 80% of total budget for this phase
var phase4Limit: nat := (maxSteps * 80) / 100;
if phase4Limit < steps {
  phase4Limit := steps;
}
if phase4Limit > maxSteps {
  phase4Limit := maxSteps;
}

while steps < phase4Limit && !insideConstrainedOut && !hasCompletedSpan
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases phase4Limit - steps
{
  var actualChunk4 := chunkSize;
  if steps + actualChunk4 > phase4Limit {
    actualChunk4 := phase4Limit - steps;
  }
  if actualChunk4 == 0 {
    break;
  }
  var cg4, stoppedOnOpen4, stoppedOnEos4, stepsUsed4 := helpers.UnconstrainedChunk(lm, prompt, generated, actualChunk4, "<<", eosToken);
  generated := cg4;
  steps := steps + stepsUsed4;
  if stoppedOnEos4 {
    break;
  }
  if stoppedOnOpen4 {
    var eg4, ei4, ec4 := helpers.EnterObservedConstrainedSpan(lm, generated);
    generated := eg4;
    insideConstrainedOut := ei4;
    currentConstrainedOut := ec4;
  }
}

// Phase 5: If we entered a span in Phase 4, close it with constrained steps
var innerSteps2: nat := 0;

while insideConstrainedOut && !hasCompletedSpan && steps < maxSteps && innerSteps2 < innerStepLimit
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases (maxSteps - steps) + (innerStepLimit - innerSteps2)
{
  var cg2, ci2, cc2, closed2 := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
  if closed2 {
    generated := cg2;
    insideConstrainedOut := ci2;
    currentConstrainedOut := cc2;
    steps := steps + 1;
    hasCompletedSpan := true;
  } else {
    var constrainedPrompt2 := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next2 := helpers.ConstrainedStep(lm, parser, constrainedPrompt2, currentConstrainedOut, eosToken);
    steps := steps + 1;
    innerSteps2 := innerSteps2 + 1;
    if next2 == eosToken {
      break;
    } else {
      var ag2, ai2, ac2 := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next2);
      generated := ag2;
      insideConstrainedOut := ai2;
      currentConstrainedOut := ac2;
    }
  }
}

// Phase 6: If still open after Phase 5, use CloseSpanWithinBudget with all remaining
if insideConstrainedOut && steps < maxSteps {
  var remaining6 := maxSteps - steps;
  var spanBudget6: nat := remaining6;
  if spanBudget6 > 150 {
    spanBudget6 := 150;
  }
  if spanBudget6 > 0 {
    var wg6, wi6, wc6 := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, spanBudget6);
    generated := wg6;
    insideConstrainedOut := wi6;
    currentConstrainedOut := wc6;
    steps := steps + spanBudget6;
    if !insideConstrainedOut {
      hasCompletedSpan := true;
    }
  }
}

// Phase 7: If no span completed at all, force one open
if !insideConstrainedOut && !hasCompletedSpan && steps + 2 <= maxSteps {
  var fg, fi, fc := helpers.OpenConstrainedSpan(lm, generated);
  generated := fg;
  insideConstrainedOut := fi;
  currentConstrainedOut := fc;
  steps := steps + 1;

  // Use ALL remaining budget to close
  if insideConstrainedOut && steps < maxSteps {
    var remainBudget7 := maxSteps - steps;
    if remainBudget7 > 0 {
      var wg7, wi7, wc7 := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, remainBudget7);
      generated := wg7;
      insideConstrainedOut := wi7;
      currentConstrainedOut := wc7;
      steps := steps + remainBudget7;
    }
  }
}

// Phase 8: Final safety - if still inside, use ALL remaining budget to close
if insideConstrainedOut && steps < maxSteps {
  var finalBudget := maxSteps - steps;
  if finalBudget > 0 {
    var wgf, wif, wcf := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, finalBudget);
    generated := wgf;
    insideConstrainedOut := wif;
    currentConstrainedOut := wcf;
    steps := steps + finalBudget;
  }
}

cost := steps;
