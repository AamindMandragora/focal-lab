// CSD_RATIONALE_BEGIN
// Analysis of all attempts:
// - Best result: attempt 4, accuracy 34.0%, syntax 84.0%
//   Uses UnconstrainedChunk(5) to find "<<", then ConfidenceGatedStep inside.
// - Attempt 6 regressed badly (22% acc, 48% syntax) due to "entered_constrained_mode_too_early":
//   The model was generating "SQL: <<" in 2 tokens, so the 4-token chunk was
//   already past the "<<" when OpenConstrainedSpan was called, resulting in
//   double "<<" or wrong span state.
// - Key diagnostic: "Tokens before first visible open: avg 2.00, median 2.00"
//   The model emits "SQL:" then "<<" as its first two tokens. So a 4-token
//   UnconstrainedChunk will stop on "<<" after 2 tokens (stoppedOnOpenSpan=true).
//   Then EnterObservedConstrainedSpan is correct. This worked in attempt 4 (5-token chunk).
//
// Root cause of attempt 6 failure vs attempt 4:
// - Attempt 4 used chunkBudget=5 and ConfidenceGatedStep -> 34% acc, 84% syntax
// - Attempt 6 used chunkBudget=4 and ConfidenceGatedStep -> 22% acc, 48% syntax
// - The difference MUST be the guidance. Attempt 6 added complex guidance about
//   "count ( * ) not count(*)" which confused the SQL parser.
// - Also "entered_constrained_mode_too_early: 50 examples" means ALL examples
//   triggered this. But in attempt 4 this was NOT a failure mode at all.
// - The actual problem: attempt 6's guidance "Use lowercase keywords and spaces
//   around operators and parentheses: count ( * ) not count(*)" caused the
//   SQL parser to reject tokens, and the constrained step forced wrong tokens.
//
// Strategy: Return to attempt 4's exact approach but with improvements:
// 1. Use UnconstrainedChunk(5) to find "<<" naturally (model emits "SQL:" + "<<")
// 2. EnterObservedConstrainedSpan if "<<" found, else OpenConstrainedSpan
// 3. Use ConfidenceGatedStep inside span (best accuracy/syntax balance)
// 4. Close span when complete
// 5. Use MINIMAL guidance - just tell model the format, don't prescribe SQL style
//
// The key insight: attempt 4 had 34% accuracy with 84% syntax. The gap to goal
// is accuracy (34% -> 42%) and syntax (84% -> 88%). Both gaps are moderate.
// The failure cluster shows "syntax_valid_semantic_mismatch=28" - the model
// generates syntactically valid but wrong SQL. This is an accuracy problem,
// not a syntax problem. We need to improve accuracy without hurting syntax.
//
// To improve accuracy: use AdaptiveConstrainedStep with validTokenGroups boosts
// instead of ConfidenceGatedStep. The validTokenGroups contain schema-relevant
// tokens that should be boosted to guide the model toward correct table/column names.
// But AdaptiveConstrainedStep (attempt 5) got worse. So stick with ConfidenceGatedStep.
//
// Alternative: The model generates wrong SQL because it doesn't know the schema.
// The validTokenGroups should contain schema tokens. Boost them with
// GroupBoostedConstrainedStep to help the model pick correct column/table names.
//
// Final strategy: attempt 4 approach + GroupBoostedConstrainedStep with validTokenGroups
// to boost schema-relevant tokens. This should improve accuracy while maintaining syntax.
// Use a narrowThreshold to only apply group boosting when there are many valid choices
// (to avoid over-constraining narrow states).
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity invariant:
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//   Pre-loop phase:
//   - Before any span is opened, insideConstrainedOut is false (or preserved from
//     caller), so the implication is vacuously true or holds by precondition.
//   - UnconstrainedChunk: generated is extended but insideConstrainedOut stays false.
//     If stoppedOnOpenSpan: EnterObservedConstrainedSpan sets currentConstrainedOut := [],
//     and parser.IsValidPrefix([]) holds by precondition. ✓
//   - If !stoppedOnOpenSpan: OpenConstrainedSpan sets currentConstrainedOut := [],
//     same argument. ✓
//   - If stoppedOnEos: we return early, invariant trivially holds. ✓
//
//   Main loop:
//   - CloseConstrainedSpan branch: sets insideConstrainedOut := false, making
//     the implication vacuously true. ✓
//   - ConfidenceGatedStep branch: by helper postcondition, returns either EOS or
//     a token t such that parser.IsValidPrefix(currentConstrainedOut + [t]) holds.
//     AppendConstrainedToken extends currentConstrainedOut by t, preserving validity. ✓
//   - GroupBoostedConstrainedStep branch: same postcondition guarantee as
//     ConfidenceGatedStep - returns EOS or a parser-valid next token. ✓
//   - !insideConstrainedOut break: implication vacuously true. ✓
//
// progress invariant (|generated| <= |generatedPrefix| + steps):
//   Pre-loop phase:
//   - UnconstrainedChunk: steps += stepsUsed (>= 1 when chunkBudget > 0),
//     generated grows by at most stepsUsed visible tokens. ✓
//   - OpenConstrainedSpan: steps += 1, generated grows by 1 ("<<"). ✓
//   - EnterObservedConstrainedSpan: steps += 0, generated unchanged. ✓
//
//   Main loop (each branch: steps += 1, |generated| grows by at most 1):
//   - CloseConstrainedSpan: generated grows by at most 1 (">>"), steps += 1. ✓
//   - ConfidenceGatedStep + AppendConstrainedToken: when next != eosToken,
//     generated grows by exactly 1; when next == eosToken, we break without
//     appending. Either way |generated| - |generatedPrefix| <= steps. ✓
//   - GroupBoostedConstrainedStep + AppendConstrainedToken: same argument. ✓
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

// Minimal guidance: just tell the model the output format.
// Avoid prescribing SQL style (spaces, lowercase) as that confuses the parser.
helpers.AppendTaskGuidance(lm, "Output exactly one SQL query. Format: SQL: <<query>>. Use only tables and columns from the schema. No explanation.");

var steps: nat := 0;

// Phase 1: Use UnconstrainedChunk to reach the opening "<<" delimiter.
// The model typically emits "SQL:" then "<<" as its first 2 tokens.
// A budget of 6 gives enough room to find "<<" while not wasting steps.
if !insideConstrainedOut && steps < maxSteps {
  var chunkBudget: nat := 6;
  if chunkBudget > maxSteps - steps {
    chunkBudget := maxSteps - steps;
  }
  if chunkBudget > 0 {
    var chunkGenerated, stoppedOnOpenSpan, stoppedOnEos, stepsUsed :=
      helpers.UnconstrainedChunk(lm, prompt, generated, chunkBudget, "<<", eosToken);
    generated := chunkGenerated;
    steps := steps + stepsUsed;
    if stoppedOnEos {
      cost := steps;
      return;
    } else if stoppedOnOpenSpan {
      // "<<" already in generated; update span state without re-appending
      var og, oi, oc := helpers.EnterObservedConstrainedSpan(lm, generated);
      generated := og;
      insideConstrainedOut := oi;
      currentConstrainedOut := oc;
    } else {
      // "<<" not emitted naturally; force it
      if steps < maxSteps {
        var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
        generated := og;
        insideConstrainedOut := oi;
        currentConstrainedOut := oc;
        steps := steps + 1;
      }
    }
  }
}

// Phase 2: Constrained SQL generation inside << >> span.
// Use GroupBoostedConstrainedStep when validTokenGroups is non-empty to boost
// schema-relevant tokens (improves accuracy), otherwise use ConfidenceGatedStep
// (preserves model's preferred token when grammar-valid).
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
    break;
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
    break;
  } else {
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := eosToken;
    if |validTokenGroups| > 0 {
      // Use group boosting to prefer schema-relevant tokens
      next := helpers.GroupBoostedConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 3.0, eosToken
      );
    } else {
      // Fall back to confidence-gated step
      var n, w := helpers.ConfidenceGatedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, eosToken
      );
      next := n;
    }
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