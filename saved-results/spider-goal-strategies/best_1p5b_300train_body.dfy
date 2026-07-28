// CSD_RATIONALE_BEGIN
// ANALYSIS (Best = 43.7%, attempts 31+):
//
// KEY DIAGNOSIS:
// - Answer is extracted as "text_fallback" from free generation output
// - Constrained span output is "valid_nonfinal_spans_only" = not used for extraction
// - All accuracy comes from the free generation phase quality
// - mode_A: 178 wrong, syntax valid, low token budget - semantic mismatches
// - mode_B: 5 wrong, syntax invalid, mid token budget
//
// CONCRETE FAILURES:
// 1. "SELECT COUNT(paragraph_id) FROM paragraphs" vs correct "select count(*) from paragraphs"
//    - The model uses COUNT(col) instead of COUNT(*)
//    - This is a common mistake for "count the number of X"
// 2. "SELECT T1.pet_type, MAX(T1.weight)...JOIN has_pet" vs "select max(weight), pettype from pets group by pettype"
//    - Model adds unnecessary JOIN
// 3. Cars: infinite nesting of IN (SELECT...) despite guidance
//    - Correct: GROUP BY...ORDER BY COUNT(*) DESC LIMIT 1
//
// WHY PREVIOUS GUIDANCE ATTEMPTS FAILED:
// - Attempt 31 (43.7%): Best - "5 rules" guidance
// - Attempt 37 (39.0%): Added concrete INTERSECT template - hurt (-4.7pp)
// - The template was TOO SPECIFIC and confused the model on non-INTERSECT cases
//
// ROOT CAUSE ANALYSIS:
// The 1.5B model is getting confused by LONG guidance with examples.
// When guidance is long, the model treats it as context to continue,
// not as instructions to follow.
//
// KEY INSIGHT: The "COUNT(*)" failure is very specific and very common.
// "Count the number of X" → COUNT(*) is the right pattern.
// Model generates COUNT(col_name) instead.
// This is likely responsible for many of the 178 wrong answers.
//
// STRATEGY: Return to best attempt (31) guidance structure but:
// 1. Add explicit COUNT(*) rule: "For 'count the number': use COUNT(*) not COUNT(col)"
// 2. Keep INTERSECT rule but remove the verbose example template
// 3. Keep GROUP BY + ORDER BY DESC LIMIT 1 rule
// 4. Keep HAVING rule
// 5. Remove unnecessary JOIN rule (already there in best)
//
// The COUNT(*) vs COUNT(col) failure is VERY tractable because:
// - It's a simple syntactic substitution
// - The model should learn from a clear rule
// - It's probably one of the most common failures in spider
//
// ALSO: The "last_name only" vs "first_name AND last_name" failure.
// For "name" in the question → usually need just one column.
// But this is harder to specify in guidance.
//
// REVISED GUIDANCE:
// Keep the 5 rules from attempt 31 but ADD COUNT(*) rule.
// Make it 6 short rules.
//
// ATTEMPT 31 GUIDANCE (reconstructed from diagnostics):
// Best attempt had: INTERSECT, GROUP BY+ORDER BY, HAVING, avoid JOINs, output only SQL
//
// NEW GUIDANCE adds:
// - Rule 0: COUNT(*) for counting rows
// - Keep other 4 rules concise
//
// ADDITIONAL STRUCTURAL IDEA:
// The "cars/model" example shows infinite nesting despite detection.
// Detection fires at selectCount >= 5. By that time, damage is done.
// What if we detect at >= 3 or >= 4?
//
// But attempt 30 tried >= 3 and got 38.7% (worse than best).
// This means some valid queries have 3+ nested SELECTs.
// Keep >= 5 threshold from best attempt.
//
// FINAL PLAN:
// 1. Updated guidance with COUNT(*) rule added prominently
// 2. Keep free generation 4/5 budget with SELECT >= 5 detection
// 3. Keep constrained grounding phase
// 4. Keep CloseSpanWithinBudget
//
// This is a minimal targeted change from the best attempt (31).
// Adding one specific rule about COUNT(*) which addresses a very common failure.
//
// Note: The COUNT(*) failure example:
// "Count the number of paragraphs" → should be COUNT(*) FROM paragraphs
// This is a fundamental SQL pattern that a small model might confuse.
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   Phase 0 (free loop): insideConstrainedOut stays false throughout (loop condition
//     includes !insideConstrainedOut, we never set it true inside). The implication
//     insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut) is vacuously
//     true. After the loop, EnterObservedConstrainedSpan sets insideConstrainedOut := true
//     and currentConstrainedOut := []. parser.IsValidPrefix([]) holds by precondition.
//   RegenerateUnitOnGroundingFailure: postcondition guarantees the returned prefix is
//     parser-valid. currentConstrainedOut := filled preserves the invariant.
//   CloseSpanWithinBudget: postcondition gives either (insideOut=false, currentOut=[])
//     making implication vacuously true, or (insideOut=true, parser.IsValidPrefix(currentOut)).
//     Both branches preserve the invariant.
//
// progress (|generated| <= |generatedPrefix| + steps):
//   Phase 0: each iteration increments steps by 1 and either breaks (no token appended,
//     |generated| unchanged) or appends one token. |generated| grows by at most 1 per step.
//     |generated| <= |generatedPrefix| + steps throughout.
//   EnterObservedConstrainedSpan: cost +0, generated unchanged, steps unchanged. Preserved.
//   RegenerateUnitOnGroundingFailure with fillBudget: generates at most fillBudget tokens.
//     |generated_after| = |stable| + |filled| <= |stable| + |currentConstrainedOut| + fillBudget
//     = |generated| + fillBudget. steps increases by fillBudget. Preserved.
//   CloseSpanWithinBudget with closeBudget = maxSteps - steps: postcondition gives
//     |generatedOut| <= |generated| + closeBudget <= |generatedPrefix| + maxSteps.
//     steps := maxSteps. Preserved. cost := steps <= maxSteps.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

// Updated guidance: adds COUNT(*) rule prominently, keeps other key rules concise.
// Targeting the COUNT(col) vs COUNT(*) failure and other common patterns.
helpers.AppendTaskGuidance(lm, "Write one SQL query. RULES: (1) COUNT rows: use COUNT(*) not COUNT(column_name). Example: 'count the number of X' -> SELECT COUNT(*) FROM table. (2) For 'which X has the most/maximum' or 'X with most Y': SELECT col FROM table GROUP BY col ORDER BY COUNT(*) DESC LIMIT 1. Never use nested IN(SELECT). (3) For 'both condition A and condition B' from same entity: INTERSECT of two SELECT...JOIN...WHERE queries. (4) For 'at least N' count: GROUP BY col HAVING COUNT(*) >= N. (5) Simple queries: SELECT col FROM table WHERE condition. Do NOT add JOINs unless data from multiple tables is required. Output only the SQL.");

// Phase 0: free generation with 4/5 of step budget (matches best attempt 31).
var freeLimit: nat := (maxSteps * 4) / 5;
if freeLimit < 2 && maxSteps >= 2 {
  freeLimit := 2;
}
if freeLimit > maxSteps - 2 && maxSteps >= 2 {
  freeLimit := maxSteps - 2;
}

while steps < freeLimit && !insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases freeLimit - steps
{
  var next := helpers.UnconstrainedStep(lm, prompt, generated);
  steps := steps + 1;
  if next == eosToken {
    break;
  }
  generated := generated + [next];
  var glen := |generated|;
  // 4-gram repetition detection
  if glen >= |generatedPrefix| + 8 {
    if generated[glen-4..glen] == generated[glen-8..glen-4] {
      break;
    }
  }
  // 3-gram repetition detection
  if glen >= |generatedPrefix| + 6 {
    if generated[glen-3..glen] == generated[glen-6..glen-3] {
      break;
    }
  }
  // Detect excessive SELECT nesting (threshold=5 from best attempt 31)
  var genStr := CSDHelpers.PrefixToString(generated);
  var selectCountUpper := CSDHelpers.CountSubstring(genStr, "SELECT");
  var selectCountLower := CSDHelpers.CountSubstring(genStr, "select");
  if selectCountUpper >= 5 || selectCountLower >= 5 {
    break;
  }
}

// Phase 1: enter constrained span silently (no visible <<).
if !insideConstrainedOut {
  generated, insideConstrainedOut, currentConstrainedOut :=
    helpers.EnterObservedConstrainedSpan(lm, generated);
}

assert insideConstrainedOut;
assert parser.IsValidPrefix(currentConstrainedOut);
assert |currentConstrainedOut| <= |generated|;
assert |generated| <= |generatedPrefix| + steps;
assert steps <= maxSteps;

// Phase 2: ground SQL identifiers using RegenerateUnitOnGroundingFailure.
// Use 1/3 of remaining budget for grounding.
if steps < maxSteps {
  var rem: nat := maxSteps - steps;
  var fillBudget: nat := rem / 3;
  if fillBudget >= 1 {
    var stable := generated[..|generated| - |currentConstrainedOut|];
    var filled := helpers.RegenerateUnitOnGroundingFailure(
      lm, parser, prompt + stable, currentConstrainedOut, eosToken, fillBudget, 3, fillBudget);
    generated := stable + filled;
    currentConstrainedOut := filled;
    steps := steps + fillBudget;
    assert |generated| <= |generatedPrefix| + steps;
  }
}

assert insideConstrainedOut;
assert parser.IsValidPrefix(currentConstrainedOut);
assert |currentConstrainedOut| <= |generated|;
assert |generated| <= |generatedPrefix| + steps;
assert steps <= maxSteps;

// Phase 3: close the constrained span within remaining budget.
if steps < maxSteps {
  var closeBudget: nat := maxSteps - steps;
  generated, insideConstrainedOut, currentConstrainedOut :=
    helpers.CloseSpanWithinBudget(
      lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget);
  steps := maxSteps;
}

assert |generated| <= |generatedPrefix| + steps;
assert steps <= maxSteps;

cost := steps;