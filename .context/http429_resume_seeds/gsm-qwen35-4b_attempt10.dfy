// CSD_RATIONALE_BEGIN
// The verification error is: decreases expression might not decrease in Phase 3 loop.
// The issue: when spanTokens < minSpanTokens, we skip the CloseSpanIfComplete call
// and go to the ConstrainedStep block. But if ConstrainedStep returns EOS and we set
// spanDone := true, we don't increment steps first - wait, we do increment steps.
// 
// Actually the real problem: when remainingBudget <= 2, we set spanDone := true
// but DON'T increment steps. So the loop condition might still be true on next
// iteration... but spanDone is true, so loop exits. Wait, that's fine.
//
// The issue is: when spanTokens < minSpanTokens AND insideConstrainedOut && !spanDone
// && steps < maxSteps, we enter the second if block. But what if steps == maxSteps
// after that second block? The loop exits. But what if after both blocks, steps
// hasn't changed and insideConstrainedOut and !spanDone are still true?
//
// Specifically: when remainingBudget <= 2 (steps >= maxSteps - 2), we set spanDone=true
// but steps is unchanged. Then loop exits because spanDone=true. OK fine.
//
// But the problematic path: spanTokens >= minSpanTokens, CloseSpanIfComplete is called
// (steps+1), closed=false, insideConstrainedOut still true, spanDone still false.
// Then: insideConstrainedOut && !spanDone && steps < maxSteps, so ConstrainedStep
// is called (steps+1). So steps increases by 2 in that iteration. OK.
//
// What about: spanTokens < minSpanTokens (skip close), then insideConstrainedOut &&
// !spanDone && steps < maxSteps, ConstrainedStep called (steps+1), next==eosToken,
// spanDone=true. steps increased by 1. Loop terminates.
//
// What about: spanTokens < minSpanTokens, then insideConstrainedOut && !spanDone &&
// steps < maxSteps, ConstrainedStep called (steps+1), next != eosToken,
// AppendConstrainedToken called (steps unchanged after that), spanTokens+1.
// steps increased by 1. Loop condition re-evaluated. OK.
//
// Actually wait - the problem is more subtle. When remainingBudget <= 2:
// we set spanDone = true, but DON'T increment steps. Then loop checks:
// steps < maxSteps (still true) && insideConstrainedOut (still true) && !spanDone (FALSE)
// So loop exits. Fine.
//
// Hmm but Dafny needs `decreases maxSteps - steps` to strictly decrease on each
// iteration. When spanDone becomes true WITHOUT steps increasing, the decreases
// measure `maxSteps - steps` doesn't decrease.
//
// Solution: use `decreases maxSteps - steps, if spanDone then 0 else 1` or similar.
// Or better: restructure so steps always increases OR the loop terminates.
//
// Simplest fix: use `decreases maxSteps - steps, if !spanDone && insideConstrainedOut then 1 else 0`
// But Dafny's decreases clause needs lexicographic pairs.
//
// Simplest structural fix: ensure every iteration either increments steps or exits.
// When remainingBudget <= 2, instead of setting spanDone, just break.
//
// Let me restructure Phase 3 to always increment steps or break on every path.
// The cleanest approach: use CloseSpanWithinBudget for the whole constrained phase,
// which avoids writing a custom inner loop entirely.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//   Phase 1 (UnconstrainedStep loop):
//     - If next == "<<": EnterObservedConstrainedSpan sets currentConstrainedOut := [].
//       parser.IsValidPrefix([]) holds by precondition. Invariant preserved.
//     - Otherwise: insideConstrainedOut unchanged (false), implication vacuous.
//
//   Phase 2 (OpenConstrainedSpan):
//     - Sets currentConstrainedOut := []. parser.IsValidPrefix([]) holds. Preserved.
//
//   Phase 3 (inner loop with ConstrainedStep + AppendConstrainedToken + CloseSpanIfComplete):
//     - CloseSpanIfComplete: if closed, insideConstrainedOut=false (vacuous). If not,
//       state unchanged, still valid by induction.
//     - ConstrainedStep returns parser-valid token; AppendConstrainedToken extends
//       currentConstrainedOut with that valid token, preserving IsValidPrefix.
//     - Every path either increments steps (guaranteeing termination) or breaks.
//
//   Phase 4 (CloseSpanWithinBudget):
//     - Postcondition: insideOut ==> parser.IsValidPrefix(currentOut). Preserved.
//
// progress: |generated| <= |generatedPrefix| + steps
//
//   Init: steps=0, generated=generatedPrefix. Trivially holds.
//
//   Phase 1: Each iteration steps += 1. Non-EOS tokens append 1 to generated.
//     So |generated| grows by at most 1 per step. Preserved.
//
//   Phase 2 (OpenConstrainedSpan): steps += 1. generated grows by exactly 1. Preserved.
//
//   Phase 3: Each iteration increments steps by exactly 1 (one helper call per path).
//     generated grows by at most 1 (AppendConstrainedToken adds 1, CloseSpanIfComplete
//     when closed adds ">>"). So |generated| <= |generatedPrefix| + steps. Preserved.
//
//   Phase 4 (CloseSpanWithinBudget): budget = maxSteps - steps.
//     Postcondition: |generatedOut| <= |generated| + budget = |generatedPrefix| + maxSteps.
//     We set steps := maxSteps after. Preserved.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

if maxSteps == 0 {
  cost := 0;
} else {
  var guidance: string := "Solve the math word problem step by step. Write all reasoning in plain text outside << >>. At the very END, place exactly one final arithmetic expression inside << >> and then STOP. Use only: variable names, numbers, +, -, *, /, //, %, (, ), int(). Do NOT use LaTeX, dollar signs, curly braces, backticks, or ** inside << >>. Do NOT open another << after closing >>.";
  helpers.AppendTaskGuidance(lm, guidance);

  var steps: nat := 0;
  var prefixBudget: nat := (maxSteps * 3) / 5;
  if prefixBudget >= maxSteps {
    prefixBudget := maxSteps - 1;
  }

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
  // Each iteration increments steps by exactly 1 (one helper call), guaranteeing termination.
  while steps < maxSteps && insideConstrainedOut
    invariant 0 <= steps <= maxSteps
    invariant lm.ValidTokensIdsLogits()
    invariant !insideConstrainedOut ==> currentConstrainedOut == []
    invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
    invariant |generated| <= |generatedPrefix| + steps
    decreases maxSteps - steps
  {
    // Try to close if span is complete
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    steps := steps + 1;
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    if closed {
      // Span closed successfully, stop generating
      break;
    } else {
      // Need to generate more tokens inside the span
      if steps < maxSteps {
        var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
        var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
        steps := steps + 1;
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
