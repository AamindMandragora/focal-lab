// CSD_RATIONALE_BEGIN
// Analysis of current situation:
// - Best result: 62.3% (attempts 25, 30) - stuck at this plateau
// - 112 syntax_valid_wrong (mode_A): semantically incorrect SQL
// - The failing examples reveal the core problems:
//   1. "SELECT dogs.age FROM dogs" instead of "SELECT avg(age) FROM dogs"
//      -> Missing aggregation functions (AVG, SUM, COUNT, MAX, MIN)
//   2. "SELECT model FROM model_list WHERE id IN (SELECT makeid...)" instead of JOIN
//      -> Wrong query structure (subquery instead of JOIN)
//   3. "SELECT students.student_id, students.first_name, ... (huge list)" instead of
//      "SELECT other_student_details FROM students ORDER BY ..."
//      -> Wrong column selection and missing ORDER BY
//
// ROOT CAUSE ANALYSIS:
// The model generates syntactically valid SQL that doesn't match the question semantics.
// The grounding phase ensures schema tokens are used correctly, but it doesn't guide
// the model toward the RIGHT SQL STRUCTURE (aggregation, JOINs, ORDER BY, etc.).
//
// Key diagnostic: "No-complete-span wrong answers: 113/300" + "answer_source=text_fallback"
// "examples_with_activity: 110/300, correct_with_activity: 68/110 = 61.8%"
// "correct_without_activity: 119/190 = 62.6%"
// Both with and without adaptive activity have similar 62% accuracy rates.
//
// The adaptive phase at 1/2 remaining budget seems to neither help nor hurt.
// The core issue is the GROUNDING PHASE generates wrong SQL structures.
//
// NEW DIAGNOSIS:
// Looking at the examples:
// 1. "SELECT dogs.age FROM dogs" - should be AVG. The model chose `age` instead of `avg(age)`.
//    The grounding check allows both since both are in the schema. SpanGrounded accepts both.
//    So grounding can't distinguish correct vs incorrect here.
// 2. The model generates valid schema-grounded SQL but semantically wrong.
//
// WHAT CAN ACTUALLY HELP:
// The problem is fundamentally that the 7B Qwen model doesn't understand Spider SQL semantics
// well enough. Our CSD strategy can't fix semantic understanding.
//
// BUT - we can try to steer the model more aggressively using:
// 1. BETTER GUIDANCE: More specific examples of correct Spider SQL patterns
// 2. RegenerateUnitOnCheckFailure with allowed units from prompt context
//    - Extract keywords from the question (AVG, COUNT, SUM, MAX, MIN, JOIN, etc.)
//    - Check that each SQL "unit" matches expected patterns
//
// LOOKING AT SPECIFIC FAILURES:
// Q: "Compute the average age of all the dogs."
// Generated: "SELECT dogs.age FROM dogs"
// Expected: "select avg(age) from dogs"
// -> The word "average" should trigger "AVG"
//
// Q: "What is the car model with the highest mpg?"
// Generated: "SELECT model FROM model_list WHERE id IN (SELECT makeid..."
// Expected: "select car_names.model from car_names join cars_data... order by mpg desc limit 1"
// -> "highest" should trigger ORDER BY ... DESC LIMIT 1
//
// Q: "List all student details in reversed lexicographical order."
// Generated: "(huge column list)"
// Expected: "select other_student_details from students order by other_student_details desc"
// -> "reversed lexicographical order" = ORDER BY ... DESC
// -> "student details" = a single column "other_student_details"
//
// The model is generating TOO MUCH (selecting all columns vs the right one) or 
// generating wrong structure (subquery vs JOIN).
//
// GUIDANCE IMPROVEMENTS:
// Current guidance (8 rules) isn't specific enough about:
// - Aggregation function mapping: "average" -> AVG, "total" -> SUM, "count" -> COUNT
// - "highest/lowest X" -> ORDER BY X DESC/ASC LIMIT 1
// - JOIN vs subquery: prefer JOIN for relationships between tables
// - SELECT * or minimal columns: prefer specific named columns, not all columns
// - "details" usually means a single column, not all columns
//
// STRUCTURAL CHANGE: Use RegenerateUnitOnCheckFailure instead of GroundingFailure
// to check that generated SQL units are semantically grounded in question keywords.
//
// ACTUALLY - looking at the tools:
// RegenerateUnitOnCheckFailure: checks if rendered unit IS IN allowedUnits (exact match)
// RegenerateUnitOnGroundingFailure: checks lm.SpanGrounded (presence in prompt context)
//
// For SQL, allowedUnits would need to include things like "SELECT", "FROM", "WHERE",
// "AVG(", etc. - hard to enumerate from question text.
//
// MOST IMPACTFUL CHANGE: Better guidance text to steer the model's understanding.
//
// Looking at what the model gets wrong:
// 1. Missing aggregation when question asks for "average", "total", "count"
// 2. Using wrong structure (subquery vs JOIN)
// 3. Selecting wrong/too many columns
//
// NEW GUIDANCE STRATEGY:
// Focus on 3 things:
// 1. Aggregation mapping: "average"->AVG, "total/sum"->SUM, "number/count of"->COUNT,
//    "maximum/highest"->MAX, "minimum/lowest"->MIN
// 2. Multi-table joins: prefer explicit JOIN over subqueries when connecting tables
// 3. Column selection: be minimal - select only what the question asks for
//
// The current guidance has 8 rules. Let me try more targeted rules:
// - Add aggregation function mapping explicitly
// - Add "when question asks for 'details' of one entity, select a single detail column"
// - Keep alias prevention rules
//
// ALTERNATIVE STRUCTURAL APPROACH:
// Instead of just guidance + grounding, add a second grounding pass after the first one
// completes. This gives the model a second chance to "fix" wrong choices via rollback.
//
// Actually - the grounding failure detection uses lm.SpanGrounded which checks if the
// content appears in the prompt context. For SQL, "AVG" might NOT appear in the prompt
// (it's a function, not a schema item), so grounding would ACCEPT wrong choices because
// neither "AVG" nor "age" would fail the grounding check (both appear in schema context).
//
// So RegenerateUnitOnGroundingFailure might not effectively distinguish "age" vs "avg(age)".
//
// KEY INSIGHT: The grounding failure mechanism might not actually be the right approach
// for semantic SQL correctness. The model needs to CHOOSE the right SQL structure.
//
// What ACTUALLY helps is:
// 1. Temperature control (lower temperature -> more deterministic, picks highest-logit tokens)
// 2. The model's internal knowledge about SQL -> guided by AppendTaskGuidance
// 3. Speculative rollouts to compare candidate SQL
//
// TEMPERATURE APPROACH:
// Use SafeTemperatureConstrainedStep with temperature=0.5 (sharper distribution)
// to make the model more deterministic and likely to follow the prompt guidance.
//
// Or use TemperatureConstrainedStep with low temperature in the main loop.
//
// Actually attempt 25 used AdaptiveConstrainedStepWithPenalties which has hard mask.
// The hard mask already selects from valid tokens. The issue is which valid token is chosen.
//
// CRITICAL NEW IDEA:
// Use SpeculativeConstrainedRollout to look ahead and detect dead ends.
// For each candidate token, speculate forward a few steps and check if the result
// is better (e.g., leads to a complete valid parse faster).
// This is expensive but more targeted.
//
// BUT: The diagnostic shows "Examples hitting max steps: 0/300" and avg 19.96 tokens.
// The model generates SHORT SQL (avg ~20 tokens). Most SQL is short.
// So budget is NOT the issue. The issue is QUALITY in those 20 tokens.
//
// REVISED APPROACH: For short SQL, speculative rollouts might be feasible.
// With maxSteps=200 and avg 20 tokens, we have LOTS of budget spare.
// We could run multiple speculative rollouts!
//
// BUT speculative rollout is complex to implement in Dafny correctly.
// Let me stick with simpler but more targeted improvements.
//
// ACTUAL NEW APPROACH:
// 1. Keep grounding phase (it works well for schema token grounding)
// 2. REPLACE adaptive phase with SpeculativeConstrainedRollout-based validation
// 3. Use the remaining budget for close
//
// Actually, let me look at what AdaptiveConstrainedStepWithPenalties does:
// "GenerateLogits, conditional BoostValidGroups, SafePenalizeTokenLogits, MaskValidNextAndEos, ChooseNextToken"
// vs ConstrainedStep: "GenerateLogits, MaskValidNextAndEos, ChooseNextToken"
//
// The ONLY difference is:
// - Conditional BoostValidGroups (boosts tokens in validTokenGroups if they're valid next)
// - SafePenalizeTokenLogits (penalizes "AS", " AS", etc.)
//
// The question is: does the adaptive activity help or hurt?
// From the diagnostic: 110 examples had activity, 68/110=61.8% correct
//                      190 without activity: 119/190=62.6% correct
// The difference is negligible (61.8% vs 62.6%). The adaptive phase is essentially neutral.
//
// SO: The adaptive phase neither helps nor hurts significantly.
// The core accuracy comes from RegenerateUnitOnGroundingFailure.
//
// HOW TO IMPROVE:
// The grounding phase generates correct schema tokens but wrong SQL structure.
// We need to GUIDE the SQL structure more.
//
// ONE MORE OBSERVATION:
// The failing example "SELECT dogs.age FROM dogs" - this is "SELECT column FROM table"
// structure which is valid but wrong. The correct is "SELECT AVG(column) FROM table".
// The model needs to understand "compute the average" = "SELECT AVG".
//
// Our guidance says "Use exact table and column names from the schema" but doesn't say
// "Map question keywords to SQL functions".
//
// FINAL STRATEGY: Enhanced guidance + second grounding pass using RegenerateUnitOnCheckFailure
// with SQL function keywords extracted from the question as allowed units.
//
// IMPLEMENTATION:
// Phase 0: Enter constrained span
// Phase 1: RegenerateUnitOnGroundingFailure (main grounding, 3/5 budget, retries=3)
// Phase 2: RegenerateUnitOnGroundingFailure (second pass with fresh context, 1/5 budget, retries=2)
// Phase 3: CloseSpanWithinBudget (rest)
// 
// Wait - the second grounding pass would use the SAME grounding mechanism. It would only
// help if the first pass left the SQL incomplete (partial valid prefix).
// After Phase 1 completes (fills budget), currentConstrainedOut is a valid prefix.
// Phase 2 continues from where Phase 1 left off.
// This is essentially just more grounding budget - same as 4/5 but split into two parts.
//
// The reason to split: Phase 2 can use updated stable prefix + updated currentConstrainedOut.
// The prompt for Phase 2 is prompt + stable2 which includes Phase 1's output.
// This means Phase 2's LM conditioning includes the already-generated SQL as context!
// This might help the LM generate CONTINUATIONS that are more semantically coherent.
//
// YES! This is the key insight: RegenerateUnitOnGroundingFailure takes `prompt` which
// includes the existing generated SQL as part of the conditioning context.
// So Phase 2 with updated stable prefix gives the LM better context for continuation.
//
// But we're ALREADY doing this in Phase 1: `var stable := generated[..|generated| - |currentConstrainedOut|]`
// And then `helpers.RegenerateUnitOnGroundingFailure(lm, parser, prompt + stable, ...)`
// This already includes the existing generated text as part of the prompt!
//
// So splitting into two passes doesn't fundamentally change the conditioning.
// The LM already sees the existing SQL as context in Phase 1.
//
// CONCLUSION: Structural changes to the phasing won't help.
// The key lever is GUIDANCE.
//
// MOST TARGETED GUIDANCE CHANGE:
// Current: 8 generic rules
// New: Add specific SQL function mappings and patterns that 7B Qwen might not know:
//   - "average/avg" -> SELECT AVG(column) 
//   - "total/sum" -> SELECT SUM(column)
//   - "count/number of" -> SELECT COUNT(*)
//   - "maximum/highest" -> ORDER BY column DESC LIMIT 1 OR SELECT MAX(column)
//   - "minimum/lowest" -> ORDER BY column ASC LIMIT 1 OR SELECT MIN(column)
//   - Never expand "details" to all columns - select the specific detail column
//   - Use JOIN for multi-table queries, not subqueries unless needed
//   - "in reversed/descending order" -> ORDER BY ... DESC
//
// I'll also try to use ConstrainedGeneration as Phase 2 instead of adaptive loop.
// ConstrainedGeneration is simpler (just hard ConstrainedStep loop) but might 
// benefit from simpler control flow with better guidance.
//
// NEW STRUCTURE:
// Phase 0: AppendTaskGuidance (enhanced with function mappings)
// Phase 1: EnterObservedConstrainedSpan
// Phase 2: RegenerateUnitOnGroundingFailure (3/5 of budget)
// Phase 3: Hard ConstrainedStep loop (1/5 of budget) - simple hard-constrained generation
// Phase 4: CloseSpanWithinBudget (1/5 of budget)
//
// The hard ConstrainedStep in Phase 3 is the simplest possible generation.
// Combined with better guidance, it should produce better SQL completions.
//
// Actually, if grounding already works well, the adaptive phase after it only
// needs to handle edge cases. Using simple ConstrainedStep (no boosting, no penalties)
// after grounding might actually be better than adaptive step.
//
// Let me try: Phase 2 = grounding (4/5), Phase 3 = CloseSpanWithinBudget (1/5).
// This removes the adaptive phase entirely.
//
// Expected result: grounding does the heavy lifting, close handles the rest.
// Without-activity rate stays similar (~62%), with-activity goes to 0 (no adaptive).
// This should be equivalent to attempt 25 but cleaner.
// Might get ~62% still.
//
// TO PUSH HIGHER: We need the guidance to work better.
// Key guidance improvements to add (in addition to current 8 rules):
// 9. Map "average/avg" to AVG(), "total/sum" to SUM(), "count/number/how many" to COUNT(*)
// 10. For "highest X" use MAX(X) in SELECT; for "lowest X" use MIN(X)
// 11. Prefer JOIN over nested subqueries for combining data from multiple tables
// 12. Do NOT expand 'details' or 'info' columns to all table columns
// 13. For ordering: "ascending/alphabetical" -> ORDER BY ... ASC; "descending/reversed" -> ORDER BY ... DESC
//
// These directly address the failure patterns seen.
//
// IMPLEMENTATION PLAN:
// 1. Update guidance to include function mappings (rules 9-13)
// 2. Keep Phase 0: EnterObservedConstrainedSpan
// 3. Keep Phase 1: RegenerateUnitOnGroundingFailure (4/5 budget, retries=3, rollback=5)
// 4. Replace Phase 2 (adaptive loop) with simple ConstrainedStep loop (1/2 remaining)
// 5. Keep Phase 3: CloseSpanWithinBudget (rest)
//
// The simple ConstrainedStep loop is better than adaptive because:
// - No boosts/penalties to interfere
// - Pure hard-constrained generation
// - Combined with better guidance, should choose better tokens
//
// RISK: Removing the AS penalty might allow more aliases.
// MITIGATION: The guidance explicitly says no aliases, so the LM should avoid them.
//
// Let me use SafeTemperatureConstrainedStep with temperature=0.7 in Phase 2 instead.
// Lower temperature makes the model MORE deterministic and likely to follow guidance.
// This might help push the accuracy higher.
//
// FINAL DECISION:
// - Enhanced guidance (13 rules, directly addressing observed failures)  
// - Phase 1: Grounding (4/5 budget, retries=3, rollback=5) [same as attempt 25]
// - Phase 2: SafeTemperatureConstrainedStep loop with T=0.7 (1/2 remaining) [new]
// - Phase 3: CloseSpanWithinBudget (rest) [same as attempt 25]
//
// The temperature 0.7 in Phase 2 makes the model more focused, using guidance better.
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// Invariant 1 (parser_validity): insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   - Initial: by precondition. Holds.
//   - Phase 0 (EnterObservedConstrainedSpan): sets currentConstrainedOut := [], insideConstrainedOut := true.
//     parser.IsValidPrefix([]) by precondition requires parser.IsValidPrefix([]). Holds.
//   - Phase 1 (RegenerateUnitOnGroundingFailure): postcondition guarantees filled is parser-valid.
//     currentConstrainedOut := filled, insideConstrainedOut stays true. Invariant preserved.
//   - Phase 2 loop:
//     * parser.IsCompletePrefix break: no state change, invariant holds trivially (still true/valid).
//     * nextTemp == eosToken break: no state change, invariant holds.
//     * AppendConstrainedToken postcondition: returns parser-valid nc; insideConstrainedOut := ni=true.
//       Invariant preserved.
//   - Phase 3 (CloseSpanWithinBudget): postcondition guarantees either !insideOut (vacuous) or
//     parser.IsValidPrefix(currentOut). Both cases satisfy the invariant.
//
// Invariant 2 (progress): |generated| <= |generatedPrefix| + steps
//   - Initial: steps=0, generated=generatedPrefix. Holds.
//   - Phase 0 (EnterObservedConstrainedSpan): cost 0, generated unchanged, steps unchanged. Preserved.
//   - Phase 1: fillBudget1 = (rem * 4) / 5 where rem = maxSteps - steps.
//     |generated_new| = |stable| + |filled| <= |generated| + fillBudget1.
//     steps := steps + fillBudget1. Invariant preserved.
//   - Phase 2 loop: |generated| <= |generatedPrefix| + steps + loopSteps invariant.
//     Each non-break, non-eos iteration: AppendConstrainedToken grows |generated| by 1,
//     loopSteps grows by 1. SafeTemperatureConstrainedStep costs 1 regardless. Preserved.
//     After loop: steps := steps + fillBudget2; loopSteps <= fillBudget2. Preserved.
//   - Phase 3: closeBudget = maxSteps - steps. |generatedOut| <= |generated| + closeBudget
//     <= |generatedPrefix| + maxSteps. steps := maxSteps. cost := maxSteps. Holds.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

// Enhanced guidance with SQL function mappings addressing observed failures
helpers.AppendTaskGuidance(lm, "Generate accurate SQL for Spider benchmark. Critical rules: (1) NEVER use table aliases or AS keyword. Write 'table.column' or just 'column', never 'table t' or 't.column'. (2) Map question keywords to SQL: 'average/avg' -> AVG(col), 'total/sum of' -> SUM(col), 'number of/count/how many' -> COUNT(*), 'maximum/highest/most' -> MAX(col) or ORDER BY col DESC LIMIT 1, 'minimum/lowest/least' -> MIN(col) or ORDER BY col ASC LIMIT 1. (3) For 'both X and Y': use INTERSECT between two SELECT statements. (4) For 'not in/does not': use NOT IN with a subquery. (5) For ordering: 'ascending/alphabetical' -> ORDER BY col ASC; 'descending/reversed/largest first' -> ORDER BY col DESC. (6) Do NOT select all columns for 'details' - select only the specific detail column mentioned. (7) Prefer JOIN over nested subqueries when combining tables. (8) Use exact table and column names from the schema. (9) For filtering: always use WHERE clause. (10) SELECT only the columns the question asks for.");

// Phase 0: Enter constrained span silently (no visible << >>)
if !insideConstrainedOut {
  generated, insideConstrainedOut, currentConstrainedOut :=
    helpers.EnterObservedConstrainedSpan(lm, generated);
}
assert insideConstrainedOut;
assert parser.IsValidPrefix(currentConstrainedOut);
assert |currentConstrainedOut| <= |generated|;
assert |generated| <= |generatedPrefix| + steps;

// Phase 1: Schema-grounded generation (4/5 of total budget)
// retries=3, rollback=5 as in best attempt 25
if steps < maxSteps {
  var rem: nat := maxSteps - steps;
  var fillBudget1: nat := (rem * 4) / 5;
  if fillBudget1 >= 1 {
    assert |currentConstrainedOut| <= |generated|;
    var stable := generated[..|generated| - |currentConstrainedOut|];
    var filled := helpers.RegenerateUnitOnGroundingFailure(
      lm, parser, prompt + stable, currentConstrainedOut, eosToken, fillBudget1, 3, 5);
    generated := stable + filled;
    currentConstrainedOut := filled;
    steps := steps + fillBudget1;
    assert |generated| <= |generatedPrefix| + steps;
  }
}
assert insideConstrainedOut;
assert parser.IsValidPrefix(currentConstrainedOut);
assert |currentConstrainedOut| <= |generated|;
assert |generated| <= |generatedPrefix| + steps;
assert steps <= maxSteps;

// Phase 2: Temperature-constrained generation (1/2 of remaining budget)
// Temperature 0.7 makes model more deterministic, better follows guidance
if steps < maxSteps {
  var rem2: nat := maxSteps - steps;
  var fillBudget2: nat := rem2 / 2;
  var loopSteps: nat := 0;

  while loopSteps < fillBudget2
    invariant 0 <= loopSteps <= fillBudget2
    invariant lm.ValidTokensIdsLogits()
    invariant insideConstrainedOut
    invariant parser.IsValidPrefix(currentConstrainedOut)
    invariant |currentConstrainedOut| <= |generated|
    invariant |generated| <= |generatedPrefix| + steps + loopSteps
    decreases fillBudget2 - loopSteps
  {
    if parser.IsCompletePrefix(currentConstrainedOut) {
      break;
    }
    var stable2 := generated[..|generated| - |currentConstrainedOut|];
    var constrainedPrompt2 := prompt + stable2;
    var nextTemp := helpers.SafeTemperatureConstrainedStep(
      lm, parser, constrainedPrompt2, currentConstrainedOut, 0.8, eosToken);
    loopSteps := loopSteps + 1;
    if nextTemp == eosToken {
      break;
    }
    var ng, ni, nc := helpers.AppendConstrainedToken(
      lm, parser, generated, currentConstrainedOut, nextTemp);
    generated := ng;
    insideConstrainedOut := ni;
    currentConstrainedOut := nc;
  }
  steps := steps + fillBudget2;
  assert |generated| <= |generatedPrefix| + steps;
}
assert insideConstrainedOut;
assert parser.IsValidPrefix(currentConstrainedOut);
assert |currentConstrainedOut| <= |generated|;
assert |generated| <= |generatedPrefix| + steps;
assert steps <= maxSteps;

// Phase 3: Close the span within remaining budget
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