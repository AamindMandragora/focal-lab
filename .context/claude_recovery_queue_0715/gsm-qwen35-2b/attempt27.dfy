// CSD_RATIONALE_BEGIN
// Root cause: The previous attempt timed out in Dafny verification (30 s) because
// it had 8 phases with nested loops and multi-variable decreases clauses such as
// `decreases maxSteps - steps + innerStepLimit - innerSteps`. Dafny's SMT solver
// hit combinatorial explosion trying to prove those combined measures.
//
// Fix: Collapse all phases into ONE while loop with `decreases maxSteps - steps`.
// Each iteration takes exactly one LM step:
//   - Outside span: UnconstrainedStep; if output ends with "<<", enter span via
//     EnterObservedConstrainedSpan (cost +0).
//   - Inside span: CloseSpanIfComplete (free check); if closed → break; else
//     ConstrainedStep + AppendConstrainedToken.
// This single linear structure eliminates all nested loops and multi-variable
// decreases, reducing the verification obligation to something Dafny handles
// trivially within the 30 s budget.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// Invariant 1 – parser_validity:
//   Outside-span branch: insideConstrainedOut stays false until
//   EnterObservedConstrainedSpan sets currentConstrainedOut := [], which satisfies
//   parser.IsValidPrefix([]) by the method precondition.
//   Inside-span, closed = true: the helper sets insideConstrainedOut := false, so
//   the implication is vacuously true and currentConstrainedOut := [] is enforced.
//   Inside-span, closed = false: CloseSpanIfComplete leaves state unchanged.
//   ConstrainedStep returns only parser-valid tokens or EOS; AppendConstrainedToken
//   appends only the parser-valid token, preserving IsValidPrefix by its
//   postcondition. The EOS path breaks without appending, leaving the invariant intact.
//
// Invariant 2 – progress (|generated| <= |generatedPrefix| + steps):
//   Outside-span branch: UnconstrainedStep costs +1 step and appends at most 1
//   token (EOS: no append). steps grows by 1 and |generated| grows by at most 1.
//   EnterObservedConstrainedSpan is +0 cost and leaves generated unchanged.
//   Inside-span, closed = true: CloseSpanIfComplete appends ">>" (1 token) and
//   counts as +1 step. After steps += 1 and generated := cg:
//   |cg| = |generated_old| + 1 <= (|generatedPrefix| + steps_old) + 1
//        = |generatedPrefix| + steps_new. ✓
//   Inside-span, closed = false: CloseSpanIfComplete is +0 cost, +0 tokens.
//   ConstrainedStep is +1 step; AppendConstrainedToken appends exactly 1 token
//   (EOS path breaks without append). |generated| grows by at most 1 while
//   steps grows by 1, so the bound is preserved.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve the problem step by step. At the very end write: The answer is <<EXPR>> where EXPR is a complete arithmetic formula using ALL relevant variable names from the problem combined with operators +, -, *, /, (, ) and integer literals. The expression must contain multiple terms joined by operators - never write just a single number or single variable name.");

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
    }
    generated := generated + [next];
    if RenderedEndsWith(generated, "<<") {
      var eg, ei, ec := helpers.EnterObservedConstrainedSpan(lm, generated);
      generated := eg;
      insideConstrainedOut := ei;
      currentConstrainedOut := ec;
    }
  } else {
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    if closed {
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := steps + 1;
      break;
    } else {
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
      steps := steps + 1;
      if next == eosToken {
        break;
      }
      var ag, ai, ac := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
      generated := ag;
      insideConstrainedOut := ai;
      currentConstrainedOut := ac;
    }
  }
}

cost := steps;
