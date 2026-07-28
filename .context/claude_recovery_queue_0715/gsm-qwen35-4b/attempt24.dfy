// CSD_RATIONALE_BEGIN
// Strategy: Use GenerateWithPrefixAndManagedSpan at 78% prefix budget (best known result:
// 22.4% accuracy, 91.8% syntax). The key insight from diagnostics is that constrained
// intervention actually REDUCES accuracy (19% with activity vs 25% without), so the goal
// is to maximize free reasoning time while ensuring syntactic validity of the final span.
//
// The main targeted improvement over the best result is stronger guidance that explicitly
// instructs the model to always wrap integer results in int(), since many failures appear
// to be semantically-correct expressions missing the int() wrapper.
//
// Guidance improvement: explicitly require int() wrapping, give concrete examples,
// forbid LaTeX/braces/**, keep expression short and simple.
//
// Parameters unchanged from best result:
// - prefixBudget = 78% of maxSteps (702/900) → model reasons freely for most of budget
// - boostAmount = 8.0, narrowThreshold = 6 (from GenerateWithPrefixAndManagedSpan)
// - cost = maxSteps (conservative upper bound satisfying postcondition)
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   GenerateWithPrefixAndManagedSpan's postcondition directly guarantees:
//   !insideConstrainedOut ==> currentConstrainedOut == [] and
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut).
//   The initialization sets these to the precondition-valid inputs, and the helper preserves them.
//   No additional branches are taken after the helper call that could violate this invariant.
//
// progress: |generated| <= |generatedPrefix| + maxSteps, cost <= maxSteps
//   GenerateWithPrefixAndManagedSpan postcondition guarantees
//   |generated| <= |generatedPrefix| + maxSteps.
//   We set cost := maxSteps, satisfying cost <= maxSteps.
//   The liveness postcondition (maxSteps > 0 ==> cost > 0 or state changed) holds because
//   when maxSteps > 0, cost = maxSteps > 0 satisfies cost > 0 directly.
//   Visible span growth is bounded by the consumed token budget since each token appended
//   to generated costs one step from the total maxSteps budget.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

if maxSteps == 0 {
  cost := 0;
} else {
  var guidance: string := "Solve this math problem step by step. At the very end, write your answer as <<expr>> where expr uses only: variable names (no braces), numbers, +, -, *, /, //, %, (, ), int(). ALWAYS wrap integer results in int(). Example: <<int(n * price + base)>> or <<int((a + b) * c / 60)>>. One <<expr>> at the end only. No LaTeX, no {braces}, no ** operator. Keep the expression short and simple.";
  helpers.AppendTaskGuidance(lm, guidance);

  var prefixBudget: nat := (maxSteps * 78) / 100;
  if prefixBudget >= maxSteps {
    prefixBudget := maxSteps - 1;
  }

  var boostAmount: real := 8.0;
  var narrowThreshold: nat := 6;

  var g, ic, cc := helpers.GenerateWithPrefixAndManagedSpan(
    lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut,
    maxSteps, prefixBudget, validTokenGroups, boostAmount, narrowThreshold, eosToken
  );
  generated := g;
  insideConstrainedOut := ic;
  currentConstrainedOut := cc;

  cost := maxSteps;
}
