// CSD_RATIONALE_BEGIN
// Strategy for GSM symbolic math problems: free-form unconstrained generation while
// watching for the "<<" delimiter that opens a constrained expression span. Inside
// the span, alternate CloseSpanIfComplete (no-op when incomplete) and ConstrainedStep.
// When the parser signals completion, CloseSpanIfComplete emits ">>" and exits the span.
// Task guidance instructs the model to wrap symbolic expressions in <<...>> with simple
// arithmetic syntax compatible with the parser grammar.
// This avoids the disallowed GenerateWithPrefixAndManagedSpan and uses only verified helpers.
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// parser_validity: (insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut))
//   - Initialization: from precondition (insideConstrained ==> parser.IsValidPrefix(currentConstrained)).
//   - Unconstrained branch: insideConstrainedOut flips to true only when next == "<<",
//     setting currentConstrainedOut := [], which satisfies parser.IsValidPrefix([])
//     by the method precondition parser.IsValidPrefix([]).
//   - Constrained branch, closed == true: CloseSpanIfComplete postcondition directly ensures
//     !insideConstrainedOut (implication vacuously true) and currentConstrainedOut == [].
//   - Constrained branch, closed == false, append: IsTokenValidNext confirms parser validity;
//     AppendConstrainedToken postcondition preserves parser.IsValidPrefix(currentConstrainedOut).
//
// progress: (|generated| <= |generatedPrefix| + steps)
//   - Initialization: |generatedPrefix| <= |generatedPrefix| + 0. ✓
//   - Unconstrained branch: steps += 1, generated grows by at most 1 (one token or EOS break). ✓
//   - Constrained branch, closed == true: CloseSpanIfComplete appends ">>" (+1 length, +1 cost);
//     steps += 1. ✓
//   - Constrained branch, closed == false: CloseSpanIfComplete is +0 length (no-op), steps += 1;
//     ConstrainedStep is the token-step cost (not a further steps increment);
//     AppendConstrainedToken appends at most 1 token; net: length grows ≤ 1, steps grows by 1. ✓
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve the math problem step by step. Wrap each key expression and your final answer in <<expr>>. Use only: numbers, variable names, +, -, *, /, //, %, (, ), int(). Example: <<int(apples * price)>>. No LaTeX, no ** operator.");

var steps: nat := 0;

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
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      generated := generated + [next];
      if next == "<<" {
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }
    }
  } else {
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    steps := steps + 1;
    if closed {
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
    } else {
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
      if next == eosToken {
        break;
      } else {
        var valid := helpers.IsTokenValidNext(parser, currentConstrainedOut, next);
        if valid {
          var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
            lm, parser, generated, currentConstrainedOut, next
          );
          generated := appendedGenerated;
          insideConstrainedOut := appendedInside;
          currentConstrainedOut := appendedCurrent;
        }
      }
    }
  }
}

cost := steps;
