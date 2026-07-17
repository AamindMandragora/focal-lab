// CSD_RATIONALE_BEGIN
// DIAGNOSIS from attempt 56 (0% accuracy, 98% syntax):
// - Primary failure: `other_observed_failure` 48/49 — the evaluator extracts the LAST `<<…>>`
//   span as the answer, but the model writes many tiny single-token spans (`<<n>>`, `<<frac_1>>`,
//   `<<frac_2>>`) with operators between them as free text. The last span is just one variable.
// - Root cause: `CloseSpanIfComplete` fires after the FIRST complete sub-expression (a single
//   identifier like `n`), producing `<<n>>` then free-text ` * <<frac_1>> * <<frac_2>>`.
//   Evaluator picks `<<frac_2>>` — just a single value, not the full `n * frac_1 * frac_2`.
// - The best attempt (40.8%) used `next == "<<"` (exact match rarely fires for space-prefixed
//   tokens), so most organic `<<n * frac_1 * frac_2>>` spans ran as free text — the correct
//   full expression passed through to the evaluator. That strategy is correct in spirit.
//
// THE FIX (building from 40.8% base):
// 1. Use `RenderedEndsWith(generated, "<<")` (as required by rules) for span detection.
// 2. Inside the constrained span: do NOT close immediately when `IsCompletePrefix` first becomes
//    true. Instead, keep generating until `|currentConstrainedOut| >= minSpanTokens` (=5) AND
//    `parser.IsCompletePrefix(currentConstrainedOut)`. This turns `<<n>>` into `<<n * frac_1 * frac_2>>`
//    by forcing the LM to continue past the first complete sub-expression.
// 3. Stronger guidance: one final `<<complete_expression>>` span, not split across multiple spans.
// 4. On EOS inside span: close with bounded `CloseSpanWithinBudget` (max 20 steps).
// 5. Budget pressure (≤2 steps): emergency close with `CloseSpanWithinBudget`.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// 1. parser_validity: `insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)`
//    Outside-span branch: `insideConstrainedOut` is false, implication vacuous. When
//    `RenderedEndsWith(generated, "<<")` fires, `EnterObservedConstrainedSpan` sets
//    `currentConstrainedOut := []`; `parser.IsValidPrefix([])` holds by precondition. ✓
//    CloseConstrainedSpan branch: sets `insideConstrainedOut := false` (implication vacuous),
//    `currentConstrainedOut := []`. ✓
//    CloseSpanWithinBudget branch (budget pressure and EOS sub-path): postcondition explicitly
//    guarantees `insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)`. ✓
//    AdaptiveConstrainedStep + AppendConstrainedToken branch: helper returns only EOS or a
//    parser-valid next token; `AppendConstrainedToken` extends `currentConstrainedOut` by that
//    one valid token, preserving `IsValidPrefix` by the hard-mask contract. ✓
//
// 2. progress: `|generated| <= |generatedPrefix| + steps`
//    Outside-span branch: `steps += 1`, `generated` grows by at most 1 token (EOS path breaks
//    without appending). `EnterObservedConstrainedSpan` is cost-0 and does not change `generated`. ✓
//    CloseConstrainedSpan branch: `steps += 1`, `generated` grows by at most 1 token (`>>`). ✓
//    Budget-pressure CloseSpanWithinBudget branch: `closeB = maxSteps - steps`; helper postcondition
//    gives `|generatedOut| <= |generated| + closeB`; after `steps := steps + closeB` we have
//    `|generated| <= |generatedPrefix| + maxSteps = |generatedPrefix| + steps`. ✓
//    AdaptiveConstrainedStep branch: `steps += 1`, `AppendConstrainedToken` appends exactly one
//    token to `generated`. `|generated|` grows by 1, `steps` grows by 1. ✓
//    EOS sub-path (after AdaptiveConstrainedStep): `steps += 1` (for the constrained step);
//    `closeB <= maxSteps - steps` and `CloseSpanWithinBudget` gives `|generatedOut| <= |generated|
//    + closeB`; after `steps += closeB`, `steps <= maxSteps`. ✓
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve step by step. At the very end write exactly ONE << >> span containing the complete arithmetic expression for the final answer, using the variable names from the problem (no curly braces). Example: 'The answer is <<n * frac_1 * frac_2>>.' Do not split the expression across multiple separate << >> spans. One complete final expression only.");

var steps: nat := 0;
var minSpanTokens: nat := 5;

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
      generated, insideConstrainedOut, currentConstrainedOut :=
        helpers.EnterObservedConstrainedSpan(lm, generated);
    }
  } else {
    // Inside a constrained span: wait for minSpanTokens before allowing close.
    if |currentConstrainedOut| >= minSpanTokens && parser.IsCompletePrefix(currentConstrainedOut) {
      // Expression is long enough and complete: close the span.
      var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := steps + 1;
    } else if maxSteps - steps <= 2 {
      // Budget pressure: close whatever we have.
      var closeB: nat := maxSteps - steps;
      var cg, ci, cc := helpers.CloseSpanWithinBudget(
        lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeB
      );
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := steps + closeB;
    } else {
      // Keep generating constrained tokens to build up the expression.
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.AdaptiveConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut,
        validTokenGroups, 4.0, 12, eosToken
      );
      steps := steps + 1;
      if next == eosToken {
        // Model wants to end: close the open span with remaining budget.
        if maxSteps > steps {
          var closeB: nat;
          if maxSteps - steps <= 20 {
            closeB := maxSteps - steps;
          } else {
            closeB := 20;
          }
          var cg, ci, cc := helpers.CloseSpanWithinBudget(
            lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeB
          );
          generated := cg;
          insideConstrainedOut := ci;
          currentConstrainedOut := cc;
          steps := steps + closeB;
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
}

cost := steps;
