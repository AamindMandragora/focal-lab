// CSD_RATIONALE_BEGIN
// The postcondition failure is:
//   maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
//   insideConstrainedOut != insideConstrained ||
//   currentConstrainedOut != currentConstrained
//
// When maxSteps > 0, we must ensure at least one of the disjuncts holds.
// The simplest fix: if maxSteps > 0, we always consume at least one step
// (cost > 0), which satisfies the postcondition.
//
// The previous attempt failed because there was a path where maxSteps > 0
// but cost could be 0 (e.g., if all branches that open the span are guarded
// by conditions that could all be false on the first iteration).
//
// Fix: ensure the loop body always executes at least once when maxSteps > 0,
// which guarantees cost > 0 after the loop (since every branch in the loop
// body increments steps by 1).
//
// Strategy: simple adaptive constrained decoding for math word problems.
// Free generation for the reasoning phase, then force open a constrained span
// for the final arithmetic expression. Use AdaptiveConstrainedStep inside
// the span to prevent repetition loops.
//
// The key invariant to satisfy progress: the loop always runs at least once
// when maxSteps > 0, so cost > 0 when maxSteps > 0.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   - Free branch (insideConstrainedOut=false): implication is vacuous.
//   - OpenConstrainedSpan: sets currentConstrainedOut:=[], which satisfies
//     parser.IsValidPrefix([]) by precondition.
//   - AdaptiveConstrainedStep returns EOS or a parser-valid token.
//     AppendConstrainedToken preserves parser.IsValidPrefix by contract.
//   - CloseConstrainedSpan: sets insideConstrainedOut:=false, making implication vacuous.
//   - RollbackConstrainedToComplete: returns a complete prefix or [].
//     Both satisfy IsValidPrefix (complete implies valid; [] is valid by precondition).
//     If we then set insideConstrainedOut:=false, implication is vacuous.
//
// progress: |generated| <= |generatedPrefix| + steps
//   - Free unconstrained step: steps+1, generated grows by at most 1 (0 on EOS).
//   - OpenConstrainedSpan: steps+1, generated grows by exactly 1 ("<<").
//   - CloseConstrainedSpan: steps+1, generated grows by at most 1 (">>").
//   - AdaptiveConstrainedStep + AppendConstrainedToken: steps+1, generated grows by 1.
//   - AdaptiveConstrainedStep + EOS (no append): steps+1, RollbackConstrainedToComplete
//     shrinks or preserves generated (no step cost), CloseConstrainedSpan adds at most 1 (steps+1).
//   - RollbackConstrainedToComplete: no steps consumed, generated shrinks or stays same.
//   Each visible token appended corresponds to exactly one step increment.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve this math word problem step by step. At the very end, write ONLY the final arithmetic expression inside << >> delimiters. Use only numbers, +, -, *, /, (, ) inside the delimiters.");

var steps: nat := 0;
// Reserve some budget for the constrained span
var constrainedReserve: nat := if maxSteps >= 50 then 40 else if maxSteps >= 10 then maxSteps / 2 else maxSteps;
var freePhaseLimit: nat := if maxSteps > constrainedReserve then maxSteps - constrainedReserve else 0;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  if !insideConstrainedOut {
    if steps >= freePhaseLimit {
      // Time to open the constrained span for the final answer
      var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
      generated := og;
      insideConstrainedOut := oi;
      currentConstrainedOut := oc;
      steps := steps + 1;
    } else {
      // Free generation for reasoning
      var next := helpers.UnconstrainedStep(lm, prompt, generated);
      steps := steps + 1;
      if next == eosToken {
        // Model wants to stop early - open constrained span if budget allows
        if steps + 2 <= maxSteps {
          var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
          generated := og;
          insideConstrainedOut := oi;
          currentConstrainedOut := oc;
          steps := steps + 1;
        } else {
          break;
        }
      } else {
        generated := generated + [next];
      }
    }
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    // Span is complete - close it
    var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + 1;
    break;
  } else {
    // Inside constrained span, not yet complete
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.AdaptiveConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, 12, eosToken);
    steps := steps + 1;
    if next == eosToken {
      // EOS inside span: rollback to complete prefix if possible
      var rg, rc := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
      generated := rg;
      currentConstrainedOut := rc;
      if parser.IsCompletePrefix(currentConstrainedOut) && steps + 1 <= maxSteps {
        var cg2, ci2, cc2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
        generated := cg2;
        insideConstrainedOut := ci2;
        currentConstrainedOut := cc2;
        steps := steps + 1;
      } else {
        insideConstrainedOut := false;
        currentConstrainedOut := [];
      }
      break;
    } else {
      var ag, ai, ac := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
      generated := ag;
      insideConstrainedOut := ai;
      currentConstrainedOut := ac;
    }
  }
}

// Post-loop cleanup: if still inside a constrained span, try to close it
if insideConstrainedOut && steps < maxSteps {
  if parser.IsCompletePrefix(currentConstrainedOut) {
    var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + 1;
  } else {
    var rg, rc := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
    generated := rg;
    currentConstrainedOut := rc;
    if parser.IsCompletePrefix(currentConstrainedOut) && steps + 1 <= maxSteps {
      var cg2, ci2, cc2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := cg2;
      insideConstrainedOut := ci2;
      currentConstrainedOut := cc2;
      steps := steps + 1;
    } else {
      insideConstrainedOut := false;
      currentConstrainedOut := [];
    }
  }
}

cost := steps;