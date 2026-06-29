// CSD_RATIONALE_BEGIN
// The verification failure is on the postcondition:
//   maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
//   insideConstrainedOut != insideConstrained || currentConstrainedOut != currentConstrained
// When maxSteps > 0, we need cost > 0 (or some state change). The loop runs at least once
// when maxSteps > 0, so steps >= 1 after the loop, and cost = steps >= 1.
// The Dafny verifier can't prove this because after the loop it sees steps could theoretically
// be 0 (the loop invariant only says 0 <= steps <= maxSteps, not steps >= 1 after any iteration).
// Fix: add a loop invariant that tracks whether we've made progress. When maxSteps > 0, after
// the first iteration steps >= 1. We can use the invariant:
//   maxSteps == 0 || steps > 0 || (generated == generatedPrefix && insideConstrainedOut == insideConstrained && currentConstrainedOut == currentConstrained)
// which states: if maxSteps > 0 and steps == 0, nothing has changed yet (only at the start).
// After the first iteration steps > 0. So at exit: maxSteps == 0 OR steps > 0 (hence cost > 0).
// This invariant holds trivially at start (steps = 0, nothing changed yet), and after any
// iteration steps >= 1 so the disjunct steps > 0 is true.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   - Unconstrained branch, next != "<<": insideConstrainedOut unchanged (false), vacuous.
//   - Unconstrained branch, next == "<<": EnterObservedConstrainedSpan sets currentConstrainedOut
//     := [], which satisfies parser.IsValidPrefix([]) by precondition.
//   - Inside span, IsCompletePrefix: CloseConstrainedSpan sets insideConstrainedOut=false, vacuous.
//   - Inside span, ConstrainedStep non-EOS: AppendConstrainedToken preserves IsValidPrefix.
//   - Inside span, ConstrainedStep EOS: we set insideConstrainedOut:=false, currentConstrainedOut:=[], vacuous.
//
// progress: |generated| <= |generatedPrefix| + steps
//   - Every branch increments steps by 1 and appends at most 1 token to generated.
//   - EnterObservedConstrainedSpan does not append tokens (0 cost, 0 growth).
//   - CloseConstrainedSpan appends at most 1 token (>>), steps += 1. OK.
//   - AppendConstrainedToken appends exactly 1 token, steps was already incremented. OK.
//   - EOS paths: steps += 1, generated unchanged or smaller (no appends after EOS). OK.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve this math word problem step by step. At the very end, write the final numeric answer as a single arithmetic expression using ONLY actual numbers and operators (+, -, *, /, (, )). Wrap it between << and >>. Example: <<5*3+2>>. Numbers only inside << >>, no variable names.");

var steps: nat := 0;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  invariant maxSteps == 0 || steps > 0 || (generated == generatedPrefix && insideConstrainedOut == insideConstrained && currentConstrainedOut == currentConstrained)
  decreases maxSteps - steps
{
  if !insideConstrainedOut {
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      generated := generated + [next];
      if next == "<<" {
        var og, oi, oc := helpers.EnterObservedConstrainedSpan(lm, generated);
        generated := og;
        insideConstrainedOut := oi;
        currentConstrainedOut := oc;
      }
    }
  } else {
    var isComplete := parser.IsCompletePrefix(currentConstrainedOut);
    if isComplete {
      var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := steps + 1;
    } else {
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
      steps := steps + 1;
      if next == eosToken {
        insideConstrainedOut := false;
        currentConstrainedOut := [];
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

cost := steps;
