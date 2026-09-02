// CSD_RATIONALE_BEGIN
// The postcondition requires that if maxSteps > 0, then at least one of:
// cost > 0, generated != generatedPrefix, insideConstrainedOut != insideConstrained,
// currentConstrainedOut != currentConstrained.
//
// The fix: when maxSteps > 0, we always execute at least one step (either
// unconstrained prime, OpenConstrainedSpan, or constrained generation).
// This ensures cost > 0 when maxSteps > 0.
//
// The previous attempt failed because Dafny couldn't prove the progress
// postcondition. The issue is that when maxSteps > 0 but we exit early
// (e.g., EOS in prime phase before opening <<), cost might be > 0 already.
// But Dafny needs to see that we always make progress.
//
// The simplest fix: ensure the loop structure guarantees that when maxSteps > 0,
// we always consume at least 1 step. Since we always try to OpenConstrainedSpan
// when steps < maxSteps && !insideConstrainedOut, and maxSteps > 0 means steps=0 < maxSteps,
// we will always call OpenConstrainedSpan (costing 1 step) unless we already
// entered constrained mode or broke out. But if we broke out (EOS), cost > 0.
//
// The key insight: The progress postcondition says:
//   maxSteps == 0 || cost > 0 || generated != generatedPrefix || ...
// When maxSteps > 0, we need cost > 0 OR some output changed.
// Since we always call at least one helper when maxSteps > 0 (the prime loop
// runs at least once OR OpenConstrainedSpan runs), cost >= 1 when maxSteps > 0.
//
// But Dafny can't verify this from the loop structure alone. We need to
// restructure so it's obvious. The simplest approach: always open the
// constrained span first (no prime phase), which guarantees cost >= 1
// when maxSteps >= 1. Then do constrained generation.
//
// Returning to the base pattern: OpenConstrainedSpan unconditionally when
// maxSteps > 0 and not already inside. This gives cost >= 1 trivially.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   1. Initially: insideConstrainedOut = insideConstrained. If false, currentConstrainedOut = [] (precondition).
//      If true, parser.IsValidPrefix(currentConstrained) holds (precondition). Invariant holds initially.
//   2. OpenConstrainedSpan: sets insideConstrainedOut := true, currentConstrainedOut := [].
//      parser.IsValidPrefix([]) holds by precondition. Invariant preserved.
//   3. ConfidenceGatedStep + AppendConstrainedToken: AppendConstrainedToken ensures
//      parser.IsValidPrefix(new currentConstrainedOut). Invariant preserved.
//   4. ConfidenceGatedStep returning EOS: no state change, invariant preserved.
//   5. CloseConstrainedSpan: sets insideConstrainedOut := false, implication vacuous. Preserved.
//
// progress:
//   OpenConstrainedSpan: steps += 1, |generated| grows by 1 ("<<").
//     |generated| = |generatedPrefix| + 1 <= |generatedPrefix| + steps. Preserved.
//   ConfidenceGatedStep + AppendConstrainedToken: steps += 1, |generated| grows by 1.
//     |generated| <= |generatedPrefix| + steps. Preserved.
//   ConfidenceGatedStep returning EOS: steps += 1, |generated| unchanged.
//     |generated| <= |generatedPrefix| + steps. Preserved.
//   CloseConstrainedSpan: steps += 1, |generated| grows by 1 (">>").
//     |generated| <= |generatedPrefix| + steps. Preserved.
//   When maxSteps > 0: either we enter the OpenConstrainedSpan branch (steps becomes 1,
//     cost = steps >= 1 > 0) or we were already insideConstrained and enter the
//     constrained loop (steps becomes >= 1). Either way cost > 0.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

helpers.AppendTaskGuidance(lm, "Write a correct SQL query that answers the question exactly. Always include WHERE clauses for all filter conditions. Use exact table and column names from the schema. Output only the SQL query.");

// Phase 1: Open constrained span if not already inside
if steps < maxSteps && !insideConstrainedOut {
  var openGenerated, openInside, openCurrent := helpers.OpenConstrainedSpan(lm, generated);
  generated := openGenerated;
  insideConstrainedOut := openInside;
  currentConstrainedOut := openCurrent;
  steps := steps + 1;
}

// Phase 2: Generate constrained SQL
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
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  } else {
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next, wasConstrained := helpers.ConfidenceGatedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut, eosToken
    );
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }
  }
}

cost := steps;

