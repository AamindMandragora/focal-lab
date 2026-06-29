    // CSD_RATIONALE_BEGIN
// Analysis of the current situation:
// - Best result: attempt 3 (59% accuracy, 98% syntax) using ConfidenceGatedStep + DeadEndDetection
// - Attempt 5 (most recent) regressed to 55% accuracy, 95% syntax
// - The regression in attempt 5 was due to guidance change (removed airport hint, removed GROUP BY hint)
//
// Key diagnostics from attempt 5:
// - "entered_constrained_mode_too_early: 98/100" - model emits "<<" as first unconstrained token
// - "constrained intervention activity: 1/100" - the constraint is almost never doing anything
// - "syntax_valid_correct: 55/100, syntax_valid_wrong: 40/100" - 40 semantic errors
// - "examples_with_activity: 1/100" - ConfidenceGatedStep hard-masks only 1 time in 100 examples
//
// The core problem: the model is generating SQL naturally (98% syntax valid) without needing
// any constraint intervention. The 40 wrong answers are semantic errors, not syntax errors.
// The constraint is providing almost zero value.
//
// The gap to goal is 7pp (55% -> 62%). We need 7 more correct answers.
//
// The 40 wrong answers are semantic mismatches. Looking at the examples:
// 1. "Kyle's friends" - model uses subquery, correct uses JOIN. Both are valid SQL.
// 2. "Spanish predominantly spoken" - model overcomplicated with subqueries.
// 3. "Series names and contents count" - model uses count(DISTINCT a, b) which is invalid.
//
// The count(DISTINCT a, b) case is interesting - it's syntax invalid AND wrong.
// The correct answer is "count(DISTINCT series_name), count(DISTINCT content)".
// This is a case where the SQL parser SHOULD reject "count(DISTINCT a, b)" but
// the model outputs it anyway. This means either:
// a) The parser accepts it (unlikely)
// b) The ConfidenceGatedStep returned it as parser-valid
// c) The DeadEndDetection didn't fire
//
// Actually, "count(DISTINCT a, b)" - in some SQL dialects this is valid (MySQL allows it).
// So the parser might accept it. But the evaluator rejects it.
//
// The "Spanish predominantly spoken" example:
// Model: complex multi-subquery
// Correct: "select countrycode, max(percentage) from countrylanguage where language = 'Spanish' group by countrycode"
// The model's answer is logically wrong (checks isofficial = 'T' which is not "predominantly spoken").
// This is a semantic reasoning error, not a syntax error.
//
// To improve semantic accuracy, we need better guidance that helps the model reason correctly.
//
// Key insight from the diagnostics:
// - The model outputs "<<" as the FIRST token (avg tokens before "<<" = 1.00)
// - This means the model sees the task prompt and immediately outputs "SQL: <<"
// - The constrained phase then generates the SQL
// - The constraint barely intervenes (1/100 examples)
//
// The semantic errors are happening because the model is reasoning incorrectly about:
// 1. Multi-table queries (using subqueries instead of JOINs or INTERSECT)
// 2. Aggregate queries (using wrong aggregation logic)
// 3. Filtering conditions (using wrong column/table names)
//
// Guidance improvements that might help:
// 1. "Use JOIN instead of subqueries when possible"
// 2. "For 'both' conditions use INTERSECT"
// 3. "For 'predominantly' use MAX(percentage)"
// 4. "Use exact column names from the schema"
//
// But guidance changes are risky - attempt 5's guidance change caused regression.
// The best guidance was attempt 3's (with airport hint + GROUP BY hint).
//
// Strategy: Return to attempt 3's EXACT guidance (the one that achieved 59%).
// Then try to improve the constrained generation to catch more errors.
//
// One structural improvement: use AdaptiveConstrainedStep instead of ConfidenceGatedStep.
// AdaptiveConstrainedStep applies group boosts when the parser state is narrow (few valid tokens).
// This could help at key decision points (e.g., after SELECT, after FROM, etc.).
//
// The validTokenGroups from the task might contain SQL keywords.
// Using AdaptiveConstrainedStep with validTokenGroups could boost correct SQL keywords.
//
// However, if validTokenGroups is empty or unhelpful, this is equivalent to ConstrainedStep.
// The AdaptiveConstrainedStep is safer than pure ConstrainedStep because it only applies
// group boosts at narrow states (narrowThreshold=12), not forcing tokens at every step.
//
// Actually, looking at the API: AdaptiveConstrainedStep uses HARD MASK (same as ConstrainedStep)
// plus group boosts. This is STRICTER than ConfidenceGatedStep. This could cause regression.
//
// The safest improvement: use attempt 3's EXACT code with attempt 3's EXACT guidance.
// Attempt 3 achieved 59%. Attempt 5 regressed to 55% by changing the guidance.
// The fix is simply to restore attempt 3's guidance.
//
// But the task says attempt 3 is already the "best result" and we need to improve beyond it.
// We can't just copy attempt 3 - we need to push to 62%.
//
// Looking at the 40 semantic errors more carefully:
// The model generates syntactically valid SQL that is logically wrong.
// The constraint can't fix logical errors - it can only enforce syntax.
//
// One approach: use TopValidCandidates to inspect the LM's top choices and
// potentially select a better one. But this is complex.
//
// Another approach: improve the guidance to guide the model toward correct SQL patterns.
// The risk is regression, but we're already 7pp below goal.
//
// Looking at the specific failures:
// 1. "Kyle's friends" - model uses subquery, correct uses JOIN
//    Guidance: "Use JOIN for multi-table queries instead of nested subqueries"
// 2. "Spanish predominantly" - model uses isofficial, correct uses max(percentage)
//    Guidance: "For 'predominantly' or 'most' language use MAX(percentage) GROUP BY"
// 3. "Series names and contents count" - model uses count(DISTINCT a, b)
//    Guidance: "For counting DISTINCT values of multiple columns, use separate COUNT(DISTINCT col)"
//
// These are very specific hints. The general principle:
// "Prefer simple JOINs over subqueries. Use INTERSECT for 'both' conditions.
//  For counting distinct values of multiple columns, use separate COUNT(DISTINCT col) for each."
//
// Let me try a more targeted guidance that:
// 1. Keeps the INTERSECT hint (useful for "both" conditions)
// 2. Keeps the NOT IN hint (useful for negation)
// 3. Keeps the ORDER BY LIMIT hint (useful for max/min)
// 4. Adds: "For counting distinct values of multiple columns, use separate COUNT(DISTINCT col)"
// 5. Adds: "Prefer JOINs over subqueries for multi-table queries"
// 6. Restores the GROUP BY HAVING hint
// 7. Removes the airport-specific hint (too specific)
//
// This is a moderate guidance change. The risk is regression on some examples
// but potential gain on others.
//
// Actually, the most impactful change based on the data:
// - 40 semantic errors, 3 syntax errors
// - The constraint barely intervenes (1/100)
// - The model's SQL is mostly correct but logically wrong
//
// The guidance change is the only lever we have to improve semantic accuracy.
// The constraint strategy itself is already optimal (ConfidenceGatedStep + DeadEndDetection).
//
// FINAL DECISION: Use attempt 3's EXACT code structure with improved guidance:
// - Restore GROUP BY HAVING hint
// - Remove airport-specific hint  
// - Add hint about separate COUNT(DISTINCT col) for multiple columns
// - Add hint about preferring JOINs
// - Keep INTERSECT and NOT IN hints
// - Keep ORDER BY LIMIT hint
//
// The guidance from attempt 3 (59%) was:
// "Generate exactly: SQL: <<YOUR_QUERY>> where YOUR_QUERY is a single valid SQL statement.
//  Use only table and column names from the schema. For filtering by airport code use
//  WHERE flights.sourceairport = 'CODE' (not airportname). For largest/smallest use
//  ORDER BY col DESC/ASC LIMIT 1. For counting grouped items use GROUP BY with HAVING.
//  For set intersection use INTERSECT. For negation use NOT IN."
//
// The attempt 5 guidance (55%) removed the airport hint and GROUP BY hint.
// This caused regression. So the GROUP BY hint was HELPING.
//
// RESTORE attempt 3's EXACT guidance. This should restore 59%.
// Then add one small improvement: "For counting distinct values of multiple columns,
// use separate COUNT(DISTINCT col) for each column."
// This targets the "series names and contents count" failure.
//
// Also: "Use JOIN instead of nested subqueries when possible."
// This targets the "Kyle's friends" failure.
//
// These additions are low-risk because they're additive hints, not replacements.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity invariant:
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//   - Unconstrained branch (!insideConstrainedOut):
//     When next == "<<", we set currentConstrainedOut := [], which satisfies
//     parser.IsValidPrefix([]) by the method precondition. insideConstrainedOut
//     becomes true with a valid empty prefix. When next != "<<", insideConstrainedOut
//     stays false, making the implication vacuous. ✓
//   - Complete prefix branch: CloseConstrainedSpan postcondition sets
//     insideConstrainedOut := false, making the implication vacuous. ✓
//   - Dead-end branch: we break without modifying insideConstrainedOut or
//     currentConstrainedOut; invariant preserved from prior iteration. ✓
//   - Active constrained branch (ConfidenceGatedStep):
//     ConfidenceGatedStep postcondition guarantees that if next != eosToken then
//     parser.IsValidPrefix(currentConstrainedOut + [next]) holds.
//     AppendConstrainedToken extends currentConstrainedOut by exactly that valid
//     token, preserving validity. ✓
//
// progress invariant (|generated| <= |generatedPrefix| + steps):
//   - Outside span: UnconstrainedStep costs 1 step; we append at most 1 token
//     (or break on EOS). |generated| grows by at most 1 while steps grows by 1. ✓
//   - Complete prefix branch: CloseConstrainedSpan costs 1 step and appends
//     ">>" (1 token). |generated| grows by 1, steps grows by 1. ✓
//   - Dead-end branch: break; no tokens appended, DeadEndDetection costs 0. ✓
//   - Active constrained branch: ConfidenceGatedStep costs 1 step; on non-EOS,
//     AppendConstrainedToken appends 1 token. |generated| grows by at most 1
//     while steps grows by 1. ✓
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Generate exactly: SQL: <<YOUR_QUERY>> where YOUR_QUERY is a single valid SQL statement. Use only table and column names from the schema. For largest/smallest use ORDER BY col DESC/ASC LIMIT 1. For counting grouped items use GROUP BY with HAVING. For set intersection use INTERSECT. For negation use NOT IN. Use JOIN instead of nested subqueries when joining multiple tables. For counting distinct values of multiple columns use separate COUNT(DISTINCT col) for each column.");

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
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  } else {
    var isDeadEnd := helpers.DeadEndDetection(parser, currentConstrainedOut, 1);
    if isDeadEnd {
      break;
    }
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
