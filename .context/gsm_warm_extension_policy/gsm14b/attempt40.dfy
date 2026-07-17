// CSD_RATIONALE_BEGIN
// Analysis of failures:
// - 32/49 examples had "final_span_invalid": the << >> block was present but
//   contained formulas/variables/expressions instead of plain numbers.
// - The root cause: when the model naturally opens "<<" during free generation,
//   the span content is NOT under parser control. The strategy only runs
//   AdaptiveConstrainedStep for spans we explicitly force open, but by the time
//   "<<" is seen via UnconstrainedStep, the LM has already "decided" the content.
//
// The best result (attempt 12, 42.9% acc, 87.8% syntax) uses OpenConstrainedSpan
// to force a final answer span. The key insight: the model generates a lot of
// intermediate "<<expr>>" spans with formulas during free reasoning, and those
// are the ones failing syntax. The LAST span (forced by us) is valid.
//
// Key problems with current approach:
// 1. When the model naturally produces "<<" during free generation, we track it
//    via EnterObservedConstrainedSpan but still run constrained decoding. However,
//    the model may produce many such spans and we force close them prematurely.
// 2. The forced final span approach works (42.9% accuracy), but we can improve it.
//
// Improvements for this attempt:
// 1. Don't use EnterObservedConstrainedSpan at all - these intermediate formula spans
//    are noise. Let the model generate them freely in unconstrained mode.
//    This means: when we see "<<" in unconstrained generation, DON'T enter
//    constrained mode. Let the entire unconstrained phase run freely.
// 2. Force a single final answer span near the end of the budget.
// 3. Inside the forced span, use AdaptiveConstrainedStep to ensure parser-valid
//    numeric content only.
// 4. Use CloseSpanWithinBudget to handle the final close gracefully.
//
// This addresses:
// - repetition_loop: the long runs of "10/10 * 10/10 *..." happen because we enter
//   constrained mode on an organic "<<" and then keep generating inside a span that
//   never closes cleanly. By NOT entering constrained mode on organic "<<", we avoid this.
// - final_span_invalid: the forced final span is under full parser control.
//
// Budget allocation:
// - maxSteps = 900, stepTokenBudget = 1
// - Free reasoning: up to 800 steps (88%)
// - Forced final answer span: 100 steps budget (12%)
// - This gives enough room for the parser to produce a valid number.
//
// The guidance is updated to encourage the model to just show work in text,
// and to NOT put formulas in << >> during reasoning (since we ignore those spans
// from CSD perspective anyway).
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity (insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)):
//
//   Unconstrained phase: insideConstrainedOut stays false throughout (we never call
//   EnterObservedConstrainedSpan or set it true from organic "<<"). The implication
//   is vacuously true. insideConstrainedOut only becomes true when we call
//   OpenConstrainedSpan, which sets currentConstrainedOut := [] and
//   parser.IsValidPrefix([]) holds by precondition.
//
//   Forced span, CloseSpanWithinBudget path: The helper postcondition guarantees
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut). Preserved.
//
//   Forced span, AdaptiveConstrainedStep + AppendConstrainedToken path:
//   AdaptiveConstrainedStep hard-masks to parser-valid next tokens plus EOS.
//   If next != eosToken, AppendConstrainedToken extends currentConstrainedOut by
//   exactly that parser-valid token, preserving IsValidPrefix. Preserved.
//
//   Forced span, CloseConstrainedSpan path: sets insideConstrainedOut := false,
//   making the implication vacuously true, and currentConstrainedOut := []. Preserved.
//
// progress (|generated| <= |generatedPrefix| + steps):
//
//   Unconstrained phase: Each iteration increments steps by 1. Either EOS breaks
//   without appending (|generated| unchanged, steps+1), or we append exactly the
//   non-"<<", non-">>" token (|generated|+1, steps+1). In either case
//   |generated| <= |generatedPrefix| + steps. Preserved.
//
//   OpenConstrainedSpan: appends "<<" (1 token), steps += 1. Preserved.
//
//   CloseSpanWithinBudget: closeBudget = maxSteps - steps. Helper guarantees
//   |generatedOut| <= |generated| + closeBudget. We set steps := maxSteps.
//   |generated| <= old|generated| + closeBudget = old|generated| + maxSteps - old_steps
//   <= |generatedPrefix| + old_steps + maxSteps - old_steps = |generatedPrefix| + maxSteps
//   = |generatedPrefix| + steps (new). Preserved.
//
//   AdaptiveConstrainedStep + AppendConstrainedToken: steps += 1, generated += 1 token.
//   |generated| grows by 1, steps grows by 1. Preserved.
//
//   CloseConstrainedSpan: steps += 1, generated may grow by 1 (">>").
//   |generated| grows by at most 1, steps grows by 1. Preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve this math problem step by step. Show your reasoning using plain text. At the very end, write your final numeric answer as <<NUMBER>> where NUMBER is a plain integer or decimal (e.g. <<42>> or <<3.5>>). Do not put formulas or variable names inside << >>.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;
var narrowThreshold: nat := 12;

// Phase 1: Free generation for reasoning (up to freeStepsTarget)
// We deliberately do NOT enter constrained mode on organic "<<" tokens.
// The model may produce formula spans during reasoning - that's fine, we ignore them.
var freeStepsTarget: nat := if maxSteps >= 10 then (maxSteps * 85) / 100 else maxSteps / 2;
// Reserve at least some steps for the forced final span
var reserveSteps: nat := if maxSteps >= 10 then maxSteps - freeStepsTarget else 1;

while steps < maxSteps && !insideConstrainedOut && steps < freeStepsTarget
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  var next := helpers.UnconstrainedStep(lm, prompt, generated);
  steps := steps + 1;
  if next == eosToken {
    break;
  } else {
    generated := generated + [next];
    // Intentionally do NOT enter constrained mode on "<<" - let formulas pass through freely
  }
}

// Phase 2: Force a constrained final answer span (if budget allows and not already in span)
if !insideConstrainedOut && steps < maxSteps && maxSteps - steps >= 2 {
  // Open a constrained span for the final answer
  var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
  generated := og;
  insideConstrainedOut := oi;
  currentConstrainedOut := oc;
  steps := steps + 1;
}

// Phase 3: Inside the forced constrained span - generate the answer
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
    // Span is complete: close it
    var cg, ci, cc := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + 1;
  } else if maxSteps - steps <= 3 {
    // Near budget end: use CloseSpanWithinBudget to finish gracefully
    var closeBudget := maxSteps - steps;
    var cg, ci, cc := helpers.CloseSpanWithinBudget(
      lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
    );
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := maxSteps;
  } else {
    // Normal constrained generation inside the span
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.AdaptiveConstrainedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut,
      validTokenGroups, 4.0, narrowThreshold, eosToken
    );
    steps := steps + 1;
    if next == eosToken {
      // EOS inside span - try to close if possible, else leave
      if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
        var cg, ci, cc := helpers.CloseConstrainedSpan(
          lm, parser, generated, currentConstrainedOut
        );
        generated := cg;
        insideConstrainedOut := ci;
        currentConstrainedOut := cc;
        steps := steps + 1;
      }
      break;
    } else {
      var ag, ai, ac := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := ag;
      insideConstrainedOut := ai;
      currentConstrainedOut := ac;
    }
  }
}

cost := steps;
