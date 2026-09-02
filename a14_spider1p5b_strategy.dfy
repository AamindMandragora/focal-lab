// CSD_RATIONALE_BEGIN
// Current best: accuracy 40.0%, syntax 97.0% (attempt 12, ConfidenceGatedStep)
// Last attempt: accuracy 35.0%, syntax 94.0% (attempt 13, same structure but different guidance)
//
// Key observations:
// 1. The structure works: 100% of examples get << >> delimiters, 98% complete spans
// 2. Primary failure mode: syntax_valid_semantic_mismatch (59/100 wrong but valid SQL)
// 3. Secondary failures: 2 timeouts (repetitive JOIN loops), 4 malformed (parser failures)
// 4. The "YOUR! QUERY" output occurs when guidance text confuses the model
//
// Analysis of what changed between attempt 12 (40%) and attempt 13 (35%):
// - Both use ConfidenceGatedStep
// - Attempt 12 guidance: "Write a concise, correct SQL query. Use only the tables and columns from the schema. Output the query directly without extra conditions."
// - Attempt 13 guidance: "Generate the SQL query that answers the question using the schema above."
// - The attempt 12 guidance performed better despite containing "YOUR" risk
//
// The "YOUR! QUERY" malformed example: The model outputs "YOUR! QUERY" as the SQL.
// This is because "<<YOUR QUERY>>" appears in the task prompt (the output contract)
// and the model echoes it. The "!" token is a parser artifact.
// This is NOT caused by our guidance - it's caused by the task prompt itself
// containing "<<YOUR QUERY>>" which the model echoes.
//
// Root cause of timeouts: The model generates repetitive SQL (e.g., repeated
// "INNER JOIN student_enrolment ON ...") because ConfidenceGatedStep allows
// the model to freely repeat patterns. The parser accepts these as valid.
//
// Root cause of semantic mismatch: The model generates wrong JOINs, wrong
// conditions, wrong column names. The model's own preferences (ConfidenceGatedStep)
// lead it astray.
//
// Strategy to improve from 40% to 42%+:
// 1. Keep ConfidenceGatedStep as the primary generation method (best accuracy so far)
// 2. Use AdaptiveConstrainedStepWithPenalties to penalize recently-generated tokens
//    This prevents the repetition loops that cause timeouts
// 3. Track the last few generated tokens and penalize them to break loops
// 4. Use the best guidance from attempt 12
//
// The key insight: the repetition problem causes 2 timeouts AND potentially
// causes semantic mismatches (the model gets stuck in a wrong pattern).
// Penalizing recently-seen tokens should break these loops.
//
// Implementation:
// - Use ConfidenceGatedStep for most tokens (best accuracy)
// - After generating a token, check if we're in a repetition pattern
// - Use AdaptiveConstrainedStepWithPenalties with the last token as penalty
//   to discourage immediate repetition
// - This should reduce timeouts and potentially improve accuracy
//
// Actually, looking more carefully at the timeout examples:
// - "INNER JOIN student_enrolment ON student_enrolment.student_id = student_enrolment.student_id"
//   repeated hundreds of times
// - This is a self-referential join that the parser allows
// - The fix: penalize the most recently generated tokens
//
// But we can't easily track "last N tokens" in Dafny without complex state.
// Instead, use AdaptiveConstrainedStepWithPenalties with validTokenGroups
// as boost groups and an empty penalty list (no penalty), but this won't help.
//
// Better approach: alternate between ConfidenceGatedStep and
// AdaptiveConstrainedStepWithPenalties. Use ConfidenceGatedStep for most tokens,
// but periodically use AdaptiveConstrainedStepWithPenalties to break patterns.
//
// Actually the simplest fix: just use ConfidenceGatedStep (best result so far)
// with the best guidance. The 2pp gap might just be noise between attempts.
// Let's try to exactly replicate attempt 12 which got 40%.
//
// Wait - attempt 12 used "Write a concise, correct SQL query. Use only the tables
// and columns from the schema. Output the query directly without extra conditions."
// and got 40%. Let's try a slightly different guidance that might help more.
//
// The "entered_constrained_mode_too_early" diagnostic is a red herring - it just
// means we open the span immediately (which is correct for our strategy).
//
// To push from 40% to 42%: try adding a stronger hint about query simplicity.
// The model tends to generate overly complex queries. A hint like
// "Use the simplest possible SQL. Avoid unnecessary JOINs." might help.
//
// Also: the 2 timeouts are costing us 2 correct answers. If we can prevent those,
// we gain 2pp. The fix: detect when we're generating a very long query and
// force close when the parser says IsCompletePrefix.
// This is already handled - we close as soon as IsCompletePrefix is true.
// The problem is the model keeps generating non-complete SQL indefinitely.
//
// The real fix for timeouts: the parser never reaches IsCompletePrefix because
// the model keeps extending the SQL. We need to force completion earlier.
// One approach: if |currentConstrainedOut| > some threshold, force EOS behavior.
// But we can't easily add arbitrary length checks in the loop without breaking
// the Dafny proof structure.
//
// Alternative: use AdaptiveConstrainedStep which has a narrowThreshold parameter.
// When the parser state has few valid next tokens (narrow), it applies boosts.
// This might help guide toward completion tokens.
//
// Best strategy: keep ConfidenceGatedStep + best guidance from attempt 12.
// The 2pp gap is small enough that it might just require the right guidance.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   1. OpenConstrainedSpan sets insideConstrainedOut := true and currentConstrainedOut := [].
//      parser.IsValidPrefix([]) holds by precondition (requires parser.IsValidPrefix([])).
//      Invariant established.
//   2. In the constrained loop, ConfidenceGatedStep returns next such that either
//      next == eosToken or parser.IsValidPrefix(currentConstrainedOut + [next]).
//      AppendConstrainedToken appends next, preserving the invariant.
//   3. CloseConstrainedSpan sets insideConstrainedOut := false, making the implication vacuous.
//   4. On EOS break: state unchanged, invariant holds trivially.
//
// progress:
//   OpenConstrainedSpan: steps += 1, |generated| grows by 1 ("<<" token).
//   ConfidenceGatedStep + AppendConstrainedToken: steps += 1, |generated| grows by 1.
//   ConfidenceGatedStep returning EOS: steps += 1, |generated| unchanged (still <= |generatedPrefix| + steps).
//   CloseConstrainedSpan: steps += 1, |generated| grows by 1 (">>").
//   All branches: |generated| - |generatedPrefix| <= steps, invariant preserved throughout.
//
// Final postcondition (cost > 0 when maxSteps > 0):
//   When maxSteps > 0 and !insideConstrained: OpenConstrainedSpan executes first,
//   steps becomes 1, so cost = 1 > 0.
//   When maxSteps > 0 and insideConstrained: loop body executes at least once, cost >= 1 > 0.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

// Use the guidance that worked best in attempt 12
helpers.AppendTaskGuidance(lm, "Write a concise, correct SQL query. Use only the tables and columns from the schema. Output the query directly without extra conditions.");

// Phase 1: Force open the constrained span (costs 1 step)
if steps < maxSteps && !insideConstrainedOut {
  var openGenerated, openInside, openCurrent := helpers.OpenConstrainedSpan(lm, generated);
  generated := openGenerated;
  insideConstrainedOut := openInside;
  currentConstrainedOut := openCurrent;
  steps := steps + 1;
}

// Phase 2: Generate constrained SQL inside the span using ConfidenceGatedStep
// This gives the model freedom to choose its preferred token when valid,
// only hard-masking when the model's choice would violate the parser.
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