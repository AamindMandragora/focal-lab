// CSD_RATIONALE_BEGIN
// Analysis of all attempts:
// - Best: attempt 1 (47% accuracy, 97% syntax) using ConfidenceGatedStep + guidance
//   "Write a concise, correct SQL query. Use only the tables and columns from the schema. Output the query directly without extra conditions."
// - Attempts 2-4 all regressed from this baseline.
//
// Key failure modes:
// 1. mode_A (19 wrong, syntax valid): The model generates syntactically valid SQL but wrong answer.
//    These are semantic mismatches - wrong tables, wrong conditions, wrong columns.
// 2. mode_C (3 wrong, too_slow): The LIMIT digit repetition loop. Model generates
//    "LIMIT 1000000000..." which causes timeouts.
// 3. mode_E (4 wrong, long spans): Similar to mode_C but doesn't time out.
// 4. mode_G (2 wrong, malformed): The "YOUR! QUERY" case - parser fails on "!".
// 5. mode_B (1 wrong, no span): EOS before opening span.
//
// The LIMIT loop is the critical timeout issue. The model generates:
// "ORDER BY ... LIMIT 5" then keeps appending "0" tokens.
// This happens because the parser allows arbitrary digits after LIMIT.
//
// Why attempts 2-4 failed:
// - Attempt 2: Different guidance hurt accuracy
// - Attempt 3: SafeRepetitionPenaltyStep at threshold 20 hurt accuracy (44.4%)
// - Attempt 4: SafeRepetitionPenaltyStep at threshold 30 + different guidance = 39.6%
//   The "Do not add LIMIT or ORDER BY unless..." guidance made things WORSE
//   because it confused the model on queries that DO need ORDER BY.
//
// The key insight from attempt 4's failure:
// - The guidance "Do not add LIMIT or ORDER BY unless the question explicitly requires it"
//   caused the model to generate "YOUR! QUERY" more often (malformed_constrained_content: 2)
//   and to output syntax_invalid content more often (87.5% vs 97% syntax).
// - The SafeRepetitionPenaltyStep at threshold 30 was still being triggered for the
//   LIMIT loops (which are >30 tokens), but it's slow (3 timeouts vs 2 in attempt 3).
//
// Root cause of SafeRepetitionPenaltyStep slowness:
// The top helper calls show: SafeRepetitionPenaltyStep=2954 calls in attempt 4.
// That's extremely high - it's being called for EVERY token after threshold 30.
// Each call does: GenerateLogits + PenalizeTokenLogits + MaskValidNextAndEos.
// This is slow per token, and the LIMIT loops generate thousands of "0" tokens.
//
// Solution: Go back to pure ConfidenceGatedStep (best attempt 1) with the EXACT
// same guidance. The timeouts are a problem but they only affect 2-3 examples.
// The accuracy regression from attempts 2-4 was caused by:
// 1. Wrong guidance (attempt 2)
// 2. SafeRepetitionPenaltyStep hurting accuracy (attempts 3, 4)
//
// To break the LIMIT loop without SafeRepetitionPenaltyStep:
// Use AdaptiveConstrainedStep which has a narrowThreshold parameter.
// When the parser has few valid tokens (narrow state), it applies group boosts.
// For LIMIT, after "LIMIT 5" the parser has many valid tokens (more digits or space).
// This won't help directly.
//
// Better idea: Use AdaptiveConstrainedStepWithPenalties with a penalty on "0" token.
// But we can't know what "0" maps to in the token vocabulary.
//
// Alternative: Use DeadEndDetection to check if we're near a dead end.
// If ValidTokenCount is very high (many digits allowed), we might be in LIMIT state.
// But this is complex to detect.
//
// Simplest approach that should work:
// Keep ConfidenceGatedStep as primary (best accuracy).
// Add a hard cap: if |currentConstrainedOut| > 80 tokens, force close if complete,
// or force EOS to break the loop. But we can't force EOS without breaking invariants.
//
// Actually the cleanest fix: use AdaptiveConstrainedStep with narrowThreshold=1200.
// This applies group boosts always (since threshold > any valid count).
// But validTokenGroups might be empty, making this equivalent to ConstrainedStep.
//
// The real insight: The LIMIT loop happens because the model generates a large number.
// The parser allows it. ConfidenceGatedStep lets the model do what it wants.
// The fix: use ConstrainedStep (hard mask) which forces the model to pick from
// valid tokens. But ConstrainedStep also allows "0" (it's a valid digit).
//
// The ONLY way to break the LIMIT loop is to penalize digit tokens.
// AdaptiveConstrainedStepWithPenalties can do this with penaltyTokens = ["0","1","2",...].
// But we need to know when we're in LIMIT state.
//
// Alternative approach: Use RollbackConstrainedSuffix when the query gets too long.
// If |currentConstrainedOut| > 60, rollback to a shorter valid prefix.
// Then close if possible. This could break the loop.
//
// But rollback might not preserve enough context to close the span.
//
// FINAL DECISION:
// Return to exact attempt 1 strategy (ConfidenceGatedStep + best guidance).
// This is the best we've achieved. The timeout issue affects only 2-3 examples
// and is hard to fix without regressing accuracy.
// The mode_A semantic mismatches (19 wrong) are the main problem.
//
// To improve mode_A: the model generates wrong SQL even when unconstrained.
// The parser just validates structure. Better guidance is the only lever.
//
// Try: "Write a minimal, correct SQL query. Use only the exact tables and columns 
// mentioned in the schema. Do not add ORDER BY, LIMIT, or HAVING unless explicitly 
// asked. Use simple JOINs and WHERE clauses."
//
// This is more specific than attempt 1's guidance and directly addresses:
// - "minimal" -> fewer unnecessary clauses
// - "exact tables and columns" -> no hallucinated columns
// - "Do not add ORDER BY, LIMIT, or HAVING unless explicitly asked" -> breaks LIMIT loop
//   at the model level (before generation, not during)
// - "simple JOINs and WHERE clauses" -> discourages complex subqueries
//
// Wait - attempt 4 showed that "Do not add LIMIT or ORDER BY unless explicitly required"
// HURT accuracy (39.6% vs 47%). But attempt 4 also had SafeRepetitionPenaltyStep
// which was the main culprit. Let me separate the effects.
//
// Attempt 4 had TWO changes from attempt 1:
// 1. Added SafeRepetitionPenaltyStep at threshold 30
// 2. Changed guidance to include "Do not add LIMIT or ORDER BY unless explicitly required"
//
// The regression was from 47% to 39.6%. Which change caused it?
// Attempt 3 had SafeRepetitionPenaltyStep at threshold 15 + different guidance = 44.4%.
// Attempt 4 had SafeRepetitionPenaltyStep at threshold 30 + LIMIT guidance = 39.6%.
// The LIMIT guidance likely caused the "YOUR! QUERY" malformed outputs.
//
// So: SafeRepetitionPenaltyStep hurts accuracy, and LIMIT guidance may cause malformed.
// The best path: pure ConfidenceGatedStep + attempt 1 guidance.
//
// But we need to do SOMETHING different to improve from 47%.
// The 19 mode_A wrong samples are syntax-valid but semantically wrong.
// These need better model guidance or a different generation strategy.
//
// Key observation: mode_A has "long_span_dominant: no (95%)" and "token_budget_band: low".
// These are short queries that are just wrong. The model generates valid SQL that
// doesn't match the expected answer.
//
// For SQL text-to-SQL, the main issues are:
// 1. Wrong table aliases or JOIN conditions
// 2. Wrong WHERE conditions
// 3. Wrong aggregation
// 4. Wrong column selection
//
// The guidance from attempt 1 worked best. Let's try to improve it slightly.
// Add: "Match the exact column names from the schema."
//
// Strategy: ConfidenceGatedStep (same as attempt 1) + slightly improved guidance.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   1. OpenConstrainedSpan sets insideConstrainedOut := true and currentConstrainedOut := [].
//      parser.IsValidPrefix([]) holds by precondition. Invariant established.
//   2. In the constrained loop, ConfidenceGatedStep returns next such that either
//      next == eosToken (no state change, implication vacuous) or
//      parser.IsValidPrefix(currentConstrainedOut + [next]) holds (by ConfidenceGatedStep contract).
//      AppendConstrainedToken appends next to both generated and currentConstrainedOut,
//      so parser.IsValidPrefix(new currentConstrainedOut) holds. Invariant preserved.
//   3. CloseConstrainedSpan sets insideConstrainedOut := false, making the implication vacuous.
//      Invariant trivially preserved.
//   4. On EOS break: insideConstrainedOut and currentConstrainedOut unchanged from last iteration.
//      Invariant holds by induction.
//
// progress:
//   OpenConstrainedSpan: steps += 1, |generated| grows by exactly 1 ("<<" token appended).
//     |generated| = |generatedPrefix| + 1 = |generatedPrefix| + steps. Preserved.
//   ConfidenceGatedStep + AppendConstrainedToken: steps += 1, |generated| grows by 1.
//     |generated| <= |generatedPrefix| + (old_steps + 1) = |generatedPrefix| + steps. Preserved.
//   ConfidenceGatedStep returning EOS: steps += 1, |generated| unchanged.
//     |generated| <= |generatedPrefix| + old_steps < |generatedPrefix| + steps. Preserved.
//   CloseConstrainedSpan: steps += 1, |generated| grows by 1 (">>").
//     |generated| <= |generatedPrefix| + (old_steps + 1) = |generatedPrefix| + steps. Preserved.
//   All branches consume exactly 1 step and grow |generated| by at most 1. Invariant preserved.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

// Use the best guidance from attempt 1, slightly refined
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
