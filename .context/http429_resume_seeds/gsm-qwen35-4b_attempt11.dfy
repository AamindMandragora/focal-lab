// CSD_RATIONALE_BEGIN
// Key failure analysis:
// 1. "final_span_unclosed": After the first valid span closes (>>), the model emits another
//    << but the strategy doesn't prevent it. The problem is that after CloseConstrainedSpan,
//    the generation continues freely and opens another << span which never closes.
// 2. "tiny_span_dominant": The constrained spans are tiny (avg 1.51 tokens valid), meaning
//    the parser accepts very short expressions but the model needs to emit a complete arithmetic
//    expression.
// 3. The main fix: after the first constrained span closes successfully, STOP generating.
//    Don't continue with free generation that might open another <<.
// 4. Also: the model outputs things like <<n * frac_1 * frac_2>><<  - after closing >>,
//    the free generation immediately opens another <<. We must break after first span closes.
// 5. Strategy: Generate freely until << appears, then constrain the span fully with
//    AdaptiveConstrainedStep (which handles narrow parser states well), close it, then STOP.
// 6. Use SafeRepetitionPenaltyStep inside the span to avoid the repetition loop failure.
// 7. Use a reasonable prefix budget to allow reasoning, then force open if needed.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//   Phase 1 (UnconstrainedStep loop, outside span):
//     - insideConstrainedOut is false throughout; implication is vacuously true.
//     - When next == "<<": EnterObservedConstrainedSpan sets currentConstrainedOut := [],
//       which satisfies parser.IsValidPrefix([]) by precondition. Invariant preserved.
//
//   Phase 2 (OpenConstrainedSpan, if not yet inside):
//     - Sets currentConstrainedOut := []. parser.IsValidPrefix([]) holds. Preserved.
//
//   Phase 3 (inner constrained loop):
//     - CloseSpanIfComplete: if closed, insideConstrainedOut=false (vacuous). If not,
//       state unchanged, still valid by induction hypothesis.
//     - SafeRepetitionPenaltyStep returns EOS or parser-valid next token.
//       AppendConstrainedToken preserves IsValidPrefix by contract.
//     - Every path increments steps by exactly 1.
//     - After closed=true, we break immediately; no further generation happens.
//
//   Phase 4 (CloseSpanWithinBudget if still open):
//     - Postcondition: insideOut ==> parser.IsValidPrefix(currentOut). Preserved.
//
// progress: |generated| <= |generatedPrefix| + steps
//
//   Init: steps=0, generated=generatedPrefix. Trivially holds.
//
//   Phase 1: Each iteration steps += 1; non-EOS tokens append exactly 1 to generated.
//     So |generated| grows by at most 1 per step. Preserved.
//
//   Phase 2 (OpenConstrainedSpan): steps += 1, generated grows by exactly 1 (the "<<" token).
//     Preserved.
//
//   Phase 3: Each iteration increments steps by exactly 1.
//     CloseSpanIfComplete: when closed, appends ">>" (1 token), steps += 1. Preserved.
//     When not closed, ConstrainedStep or SafeRepetitionPenaltyStep costs 1 step,
//     AppendConstrainedToken adds 1 token. Preserved.
//
//   Phase 4 (CloseSpanWithinBudget): budget = maxSteps - steps.
//     Postcondition: |generatedOut| <= |generated| + budget = |generatedPrefix| + maxSteps.
//     We set steps := maxSteps. Preserved.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

if maxSteps == 0 {
  cost := 0;
} else {
  var guidance: string := "Solve the math word problem step by step. Write all reasoning outside << >>. At the very END, place exactly one final arithmetic expression inside << >> using only: variable names, numbers, +, -, *, /, //, %, (, ), int(). Then STOP immediately. Do NOT open another << after closing >>.";
  helpers.AppendTaskGuidance(lm, guidance);

  var steps: nat := 0;
  // Allow most of the budget for reasoning, but reserve enough for the constrained span
  var prefixBudget: nat := (maxSteps * 7) / 10;
  if prefixBudget == 0 { prefixBudget := 1; }
  if prefixBudget >= maxSteps { prefixBudget := maxSteps - 1; }

  // Phase 1: Free generation until "<<" is observed or prefixBudget exhausted
  while steps < prefixBudget && !insideConstrainedOut
    invariant 0 <= steps <= maxSteps
    invariant lm.ValidTokensIdsLogits()
    invariant !insideConstrainedOut ==> currentConstrainedOut == []
    invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
    invariant |generated| <= |generatedPrefix| + steps
    decreases prefixBudget - steps
  {
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      generated := generated + [next];
      if next == "<<" {
        var g2, ic2, cc2 := helpers.EnterObservedConstrainedSpan(lm, generated);
        generated := g2;
        insideConstrainedOut := ic2;
        currentConstrainedOut := cc2;
      }
    }
  }

  // Phase 2: Force open a constrained span if not already inside one
  if !insideConstrainedOut && steps < maxSteps {
    var g2, ic2, cc2 := helpers.OpenConstrainedSpan(lm, generated);
    generated := g2;
    insideConstrainedOut := ic2;
    currentConstrainedOut := cc2;
    steps := steps + 1;
  }

  // Phase 3: Constrained generation inside the span
  // Each iteration increments steps by exactly 1.
  // We break immediately after the span closes successfully.
  var spanClosed: bool := false;
  while steps < maxSteps && insideConstrainedOut && !spanClosed
    invariant 0 <= steps <= maxSteps
    invariant lm.ValidTokensIdsLogits()
    invariant !insideConstrainedOut ==> currentConstrainedOut == []
    invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
    invariant |generated| <= |generatedPrefix| + steps
    decreases maxSteps - steps
  {
    // Try to close if complete
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    steps := steps + 1;
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    if closed {
      spanClosed := true;
      // STOP: do not generate any more tokens after closing the span
    } else {
      // Generate one more constrained token if budget allows
      if steps < maxSteps {
        var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
        // Use repetition penalty to avoid loops, adaptive for narrow states
        var next := helpers.SafeRepetitionPenaltyStep(lm, parser, constrainedPrompt, currentConstrainedOut, generated, 2.0, eosToken);
        steps := steps + 1;
        if next == eosToken {
          // EOS inside span: try to close whatever we have
          // (we'll fall through to Phase 4)
        } else {
          var ag, ai, ac := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
          generated := ag;
          insideConstrainedOut := ai;
          currentConstrainedOut := ac;
        }
      }
    }
  }

  // Phase 4: If span still open, use CloseSpanWithinBudget with remaining budget
  if insideConstrainedOut && steps < maxSteps {
    var closeBudget: nat := maxSteps - steps;
    var cg, ci, cc := helpers.CloseSpanWithinBudget(
      lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
    );
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := maxSteps;
  }

  cost := steps;
}
