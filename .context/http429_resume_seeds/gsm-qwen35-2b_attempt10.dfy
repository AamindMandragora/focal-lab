// CSD_RATIONALE_BEGIN
// Deep analysis of failures:
// 1. Accuracy is 2.0% despite 98% syntax rate - the model produces correct syntax but wrong math.
// 2. The core problem is the model generates reasoning with WRONG math, then wraps that wrong
//    answer in <<>>. The constrained span correctly captures whatever the model wrote, but the
//    model's reasoning is flawed.
//
// Key observations from rollouts:
// - Model generates verbose LaTeX-style reasoning (avg 177 tokens before "<<")
// - The constrained content is often repetitive garbage: "2/n*1/n*1/n*..." or "total - n1 - n2+1- n2+1-..."
// - The model tries to write the expression but then the constrained generation loops
//
// Root cause of repetition loops: The model writes a reasonable expression like "total - n1 - n2",
// then inside the constrained span it keeps appending "+1- n2+1-..." because the constrained
// generation doesn't stop cleanly. The CloseSpanWithinBudget should handle this but the
// model is entering the span WHILE STILL MID-EXPRESSION, so it keeps generating.
//
// The repetition happens because:
// 1. Model writes "The final answer is <<"
// 2. Inside the span, it generates "total - n1 - n2" which is valid
// 3. But then it keeps appending more operators and tokens because parser allows continuation
// 4. This creates the loop pattern
//
// FIX: Use RepetitionPenaltyStep inside the constrained span to prevent the model from
// repeating the same tokens. Also use a shorter span budget to force early closure.
//
// Key insight: The model is producing "total - n1 - n2+1- n2+1..." because the parser
// accepts any arithmetic expression and the model keeps adding "+1-n2" repetitively.
// We need to penalize recently-used tokens inside the span.
//
// Also, from the "comet" example: model writes "relative_age*1000000000000000000000000000000000000000000000"
// - this is because the model started going off-rails before the "<<". The problem is
// the model's unconstrained reasoning is wrong.
//
// The REAL fix needed: We can't fix the model's reasoning quality directly.
// But we can try:
// 1. Use a MUCH shorter free generation phase (model tends to get it right quickly or not at all)
// 2. Use RepetitionPenaltyStep inside the span to prevent repetition loops
// 3. Use TemperatureConstrainedStep with lower temperature to reduce randomness in span
// 4. Use AdaptiveConstrainedStepWithPenalties to penalize tokens that appear in currentConstrained
//
// Strategy:
// Phase 1: Generate freely for up to 600 tokens, watching for "<<" with UnconstrainedChunk
// Phase 2: Inside span, use RepetitionPenaltyStep to prevent loops, with CloseSpanIfComplete
//          at each step, limited to 60 tokens
// Phase 3: If no span completed, force open one and use same strategy
// Phase 4: Any remaining open span, close with CloseSpanWithinBudget
//
// The repetition penalty inside constrained generation should prevent "+1-n2+1-n2..." loops
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   Phase 1 (UnconstrainedChunk loop): insideConstrainedOut remains false until stoppedOnOpenSpan.
//     When stoppedOnOpenSpan, generated already ends with "<<" and we call EnterObservedConstrainedSpan
//     which sets currentConstrainedOut := [] (valid by parser.IsValidPrefix([]) precondition).
//   Phase 2 (constrained step loop): We use RepetitionPenaltyStep which is a ConstrainedStep variant
//     that returns EOS or a parser-valid next token. AppendConstrainedToken preserves validity.
//     CloseSpanIfComplete either closes (insideConstrainedOut := false, vacuous implication) or
//     is a no-op (validity unchanged).
//   Phase 3 (forced open): OpenConstrainedSpan sets currentConstrainedOut := [] (valid).
//     Same constrained step loop preserves validity.
//   Phase 4 (CloseSpanWithinBudget): postcondition guarantees invariant.
//
// progress (|generated| <= |generatedPrefix| + steps):
//   Phase 1: UnconstrainedChunk with actualChunk tokens returns stepsUsed <= actualChunk and
//     |generatedOut| <= |generated_before| + stepsUsed. We add stepsUsed to steps.
//   Phase 2: RepetitionPenaltyStep costs +1, CloseSpanIfComplete costs +1 (when it closes) or +0.
//     Each loop iteration increments steps by 1 (RepetitionPenaltyStep) + 1 (CloseSpanIfComplete).
//     Wait - CloseSpanIfComplete costs 1 when closed, 0 when not. But we call it BEFORE the
//     constrained step, so it costs at most 1. Then RepetitionPenaltyStep costs 1.
//     So each inner loop iteration costs at most 2 steps and appends at most 2 tokens. 
//     We track steps carefully to stay within maxSteps.
//   Phase 3: OpenConstrainedSpan costs 1 step. Same constrained step loop as Phase 2.
//   Phase 4: CloseSpanWithinBudget with budget b costs <= b steps and appends <= b tokens.
//     We add exactly b to steps.
//   All phases maintain |generated| <= |generatedPrefix| + steps <= |generatedPrefix| + maxSteps.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve step by step. At the very end, write exactly: The answer is <<EXPR>> where EXPR is a simple arithmetic expression using only the variable names from the problem, numbers, and operators +, -, *, /, (, ). Keep the expression as short as possible. No LaTeX, no {}, no **, no $, no fractions notation, no backslashes. Example: The answer is <<n * price - discount>>";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;
var hasCompletedSpan: bool := false;
var chunkSize: nat := 40;

// Phase 1: Unconstrained generation in chunks watching for "<<" opening delimiter
// Allow up to 75% of budget for free reasoning
var phase1Limit: nat := (maxSteps * 75) / 100;
if phase1Limit == 0 && maxSteps > 0 {
  phase1Limit := 1;
}

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

// Phase 2: If inside a constrained span, generate with repetition penalty to avoid loops
// Use step-by-step with CloseSpanIfComplete at each step
var innerStepLimit: nat := 60;
var innerSteps: nat := 0;

while insideConstrainedOut && !hasCompletedSpan && steps < maxSteps && innerSteps < innerStepLimit
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps + innerStepLimit - innerSteps
{
  // Try to close if complete
  var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
  if closed {
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + 1;
    hasCompletedSpan := true;
  } else {
    // Use repetition penalty step to avoid loops
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.RepetitionPenaltyStep(lm, parser, constrainedPrompt, currentConstrainedOut, generated, 3.0, eosToken);
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

// Phase 3: If span still open after inner loop, use CloseSpanWithinBudget to close it
if insideConstrainedOut && steps < maxSteps {
  var spanBudget: nat := 30;
  var remaining := maxSteps - steps;
  if spanBudget > remaining {
    spanBudget := remaining;
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

// Phase 4: If no span completed, continue free generation for a bit more
while steps < phase1Limit && !insideConstrainedOut && !hasCompletedSpan
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases phase1Limit - steps
{
  var next := helpers.UnconstrainedStep(lm, prompt, generated);
  steps := steps + 1;
  if next == eosToken {
    break;
  }
  generated := generated + [next];
  if next == "<<" {
    var eg, ei, ec := helpers.EnterObservedConstrainedSpan(lm, generated);
    generated := eg;
    insideConstrainedOut := ei;
    currentConstrainedOut := ec;
  }
}

// Phase 5: If we entered a span in Phase 4, close it with repetition penalty steps
var innerSteps2: nat := 0;

while insideConstrainedOut && !hasCompletedSpan && steps < maxSteps && innerSteps2 < innerStepLimit
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps + innerStepLimit - innerSteps2
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
    var next2 := helpers.RepetitionPenaltyStep(lm, parser, constrainedPrompt2, currentConstrainedOut, generated, 3.0, eosToken);
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

// Phase 6: If still open, use CloseSpanWithinBudget
if insideConstrainedOut && steps < maxSteps {
  var spanBudget2: nat := 30;
  var remaining2 := maxSteps - steps;
  if spanBudget2 > remaining2 {
    spanBudget2 := remaining2;
  }
  if spanBudget2 > 0 {
    var wg2, wi2, wc2 := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, spanBudget2);
    generated := wg2;
    insideConstrainedOut := wi2;
    currentConstrainedOut := wc2;
    steps := steps + spanBudget2;
    if !insideConstrainedOut {
      hasCompletedSpan := true;
    }
  }
}

// Phase 7: If no span completed, force one open and close within remaining budget
if !insideConstrainedOut && !hasCompletedSpan && steps < maxSteps {
  if steps + 2 <= maxSteps {
    var fg, fi, fc := helpers.OpenConstrainedSpan(lm, generated);
    generated := fg;
    insideConstrainedOut := fi;
    currentConstrainedOut := fc;
    steps := steps + 1;

    // Use repetition penalty steps inside the forced span
    var innerSteps3: nat := 0;
    var innerLimit3: nat := 40;

    while insideConstrainedOut && !hasCompletedSpan && steps < maxSteps && innerSteps3 < innerLimit3
      invariant 0 <= steps <= maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant !insideConstrainedOut ==> currentConstrainedOut == []
      invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
      invariant |generated| <= |generatedPrefix| + steps
      decreases maxSteps - steps + innerLimit3 - innerSteps3
    {
      var cg3, ci3, cc3, closed3 := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
      if closed3 {
        generated := cg3;
        insideConstrainedOut := ci3;
        currentConstrainedOut := cc3;
        steps := steps + 1;
        hasCompletedSpan := true;
      } else {
        var constrainedPrompt3 := prompt + generated[..|generated| - |currentConstrainedOut|];
        var next3 := helpers.RepetitionPenaltyStep(lm, parser, constrainedPrompt3, currentConstrainedOut, generated, 3.0, eosToken);
        steps := steps + 1;
        innerSteps3 := innerSteps3 + 1;
        if next3 == eosToken {
          break;
        } else {
          var ag3, ai3, ac3 := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next3);
          generated := ag3;
          insideConstrainedOut := ai3;
          currentConstrainedOut := ac3;
        }
      }
    }

    // Final close attempt with budget
    if insideConstrainedOut && steps < maxSteps {
      var remainBudget := maxSteps - steps;
      if remainBudget > 30 {
        remainBudget := 30;
      }
      if remainBudget > 0 {
        var wg3, wi3, wc3 := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, remainBudget);
        generated := wg3;
        insideConstrainedOut := wi3;
        currentConstrainedOut := wc3;
        steps := steps + remainBudget;
      }
    }
  }
}

// Phase 8: Final safety - if still inside, use remaining budget to close
if insideConstrainedOut && steps < maxSteps {
  var finalBudget := maxSteps - steps;
  if finalBudget > 0 {
    var wg4, wi4, wc4 := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, finalBudget);
    generated := wg4;
    insideConstrainedOut := wi4;
    currentConstrainedOut := wc4;
    steps := steps + finalBudget;
  }
}

cost := steps;
