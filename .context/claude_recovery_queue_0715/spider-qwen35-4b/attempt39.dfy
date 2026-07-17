// CSD_RATIONALE_BEGIN
// Analysis of the situation:
// - Best result: 49.7% accuracy, 93.7% syntax (attempts 9, 37)
// - Previous attempt (38) regressed badly with minSqlLength=8 hack: 18.0% accuracy, 38.3% syntax
// - The minimum length constraint caused 184 "wrong_after_constrained_activity" failures
//   because forcing longer generation produces syntax errors and wrong answers
//
// The 49.7% best is using: 5-step preamble + OpenConstrainedSpan + GroupBoostedConstrainedStep(6.0)
// + CloseSpanWithinBudget fallback.
//
// Key diagnostics from the best attempt:
// - 132 syntax_valid_semantic_mismatch: correct syntax but wrong SQL
// - 18 wrong_after_constrained_activity
// - 1 token_budget_exhausted
//
// Looking at failure examples:
// 1. "SELECT s.name " - model generates overly short/simple SQL without WHERE clauses
// 2. "SELECT d.note FROM death d JOIN ship>>" - model uses JOIN when not needed, uses aliases
// 3. Repetition loop in long complex queries
// 4. "SELECT s.first_name...!": syntax error with "!" at end
//
// The evaluator metrics show:
// - "Examples with visible <<: 0/300" - << is NOT visible to evaluator
// - "Examples with unmatched visible close: 103/300" - >> appears without <<
// - "Answer extraction source: text_fallback 0/300" and "none 0/300" BUT
//   "has_extracted_answer: yes" for wrong samples (from cluster constant)
//   And "hidden_or_task_extractor 0/300" for wrong
//   Wait - the answer IS being extracted somehow (0 "none 0/300" wrong)
//   Actually "none 0/300" means 0 have no extracted answer, meaning all have extracted answers
//
// The answer is extracted via the hidden/task extractor which reads the constrained span.
// The constrained span contains the SQL (e.g., "SELECT s.name").
//
// From attempt 38's feedback:
// - "answer_source: text_fallback" listed as constant across all wrong clusters
// - But "Answer extraction source: text_fallback 0/300" 
//   This is confusing. The cluster says text_fallback but the summary says 0. 
//   Actually looking more carefully at attempt 38: the failure was 18.0% accuracy,
//   38.3% syntax. This was because minSqlLength forced models to generate MORE tokens
//   past valid completions, producing syntax errors.
//
// Going back to the best approach (attempts 9/37: 49.7%, 93.7% syntax):
// The main remaining failure is semantic mismatch. To improve:
//
// 1. GUIDANCE: A more informative guidance that shows the model HOW to use the schema.
//    Current: "Output SQL: <<query>> where the query is a valid SQL SELECT statement..."
//    Better: Include explicit instruction about not using aliases, using exact column names.
//
// 2. REPETITION: The repetition loop needs to be addressed. Currently there's 1 repetition_loop
//    example in best attempts. The approach: track tokens seen in currentConstrainedOut
//    and penalize high-frequency ones using PenalizedConstrainedStep.
//    BUT: PenalizedConstrainedStep requires "forall t :: t in tokensToPenalize ==> t in lm.Tokens"
//    which is hard to prove. SafeBoostedConstrainedStep with negative amount is invalid (amount
//    should be positive for boost). 
//    Alternative: We CAN call helpers.SafeBoostTokenLogits with a negative amount since
//    the doc says "adds to existing logits" and doesn't specify positive-only.
//    Then call ConstrainedStep (separate LM call needed after boost).
//    But SafeBoostTokenLogits is a logit-shaping op that needs a preceding GenerateLogits...
//    Actually GroupBoostedConstrainedStep calls GenerateLogits internally first.
//    If we want to apply a penalty before the step, we'd need to call GenerateLogits first,
//    then apply SafeBoostTokenLogits with negative amount, then ConstrainedStep would call
//    GenerateLogits again. That doesn't work.
//    
// 3. ConfidenceGatedStep: Use this for most steps, only apply hard parser control when
//    the model's top choice is invalid. This gives the model more "freedom" to pick
//    tokens that are correct semantically while the parser catches actual errors.
//
// 4. TopValidCandidates + LogitGap: Inspect top candidates to make informed choices.
//
// Key insight from the examples:
// - The model uses table aliases (s., a., d.) which may not match the exact schema
// - The model generates table names that don't match schema exactly (cars vs cars_data)
// - These are semantic/grounding errors
//
// The best available fix: Use ConfidenceGatedStep instead of GroupBoostedConstrainedStep.
// ConfidenceGatedStep allows the model to freely pick when its top choice is parser-valid,
// and only applies hard masking when needed. This might reduce the "semantic deviation"
// caused by the boost artificially steering toward schema tokens in the wrong places.
//
// Wait - GroupBoostedConstrainedStep boosts validTokenGroups which are schema-related.
// If validTokenGroups contains schema tokens (table names, column names), boosting them
// helps the model use the right schema. But the boost might be causing issues when
// schema tokens appear at wrong syntactic positions.
//
// Looking at example: "SELECT d.note FROM death d JOIN ship" - the model used "death d" 
// (alias "d") and "JOIN ship" - both wrong for the query "notes of death events with 
// substring 'East'" (should be simple WHERE LIKE). The model chose JOIN unnecessarily.
//
// The model's natural tendency (without boosts) might produce better results for
// certain queries. GroupBoosted forces schema tokens but may push the model toward
// unnecessarily complex queries.
//
// BEST STRATEGY: Keep the best approach (attempts 9/37) EXACTLY, but try to improve
// by adding one targeted change. The safest targeted change that could help:
//
// Use ConfidenceGatedStep AFTER the first N tokens of SQL generation, and 
// GroupBoostedConstrainedStep for the first N tokens (structural tokens like SELECT, FROM).
// This hybrid might work but is complex.
//
// SIMPLEST IMPROVEMENT: Change guidance only. The guidance "Output SQL: <<query>>" is
// already clear. But let's try a more SQL-focused guidance:
// "Write the SQL query directly using exact schema column and table names. 
//  Do not use table aliases. Use WHERE instead of JOIN when possible."
//
// Actually the instruction "Do not use table aliases" might cause syntax errors since
// SQL allows aliases. Let me not include that.
//
// Better guidance: "Write a SQL SELECT query. Use exact table and column names from 
// the database schema provided above. Answer only the specific question asked."
//
// ANOTHER OPTION: Use ConstrainedStep (pure hard-masking, no boosts) to remove the
// boost interference, then compare against GroupBoostedConstrainedStep.
// Pure ConstrainedStep was attempted in early attempts but got lower accuracy.
//
// CONCLUSION: The best strategy is:
// 1. Keep exact structure of best attempt (9/37)
// 2. Try ConfidenceGatedStep instead of GroupBoostedConstrainedStep
//    - ConfidenceGatedStep lets the model use its preferred token when parser-valid
//    - This gives more semantic freedom while maintaining parser validity
//    - May help with "wrong table/column" issues since model's unconstrained preference
//      might actually be more semantically correct
//    - Risk: might reduce syntax correctness (currently 93.7%)
//
// Let me reconsider. GroupBoostedConstrainedStep at 6.0 is working well (93.7% syntax).
// The semantic errors are not fixable by changing the decoding strategy - they're 
// fundamental to the model's knowledge.
//
// What CAN be fixed:
// 1. The repetition loop: 1 example, very rare. Not worth complex changes.
// 2. The 18 "wrong_after_constrained_activity" - unclear what these are
//    Could be cases where span closes prematurely or model generates invalid SQL format
//
// Let me look at attempt 38 regression: It had 61 syntax_valid_wrong (same order as best)
// but 184 wrong_after_constrained_activity. So minSqlLength HURT the 18 wrong_after_constrained
// cases and caused ~166 more failures. The approach must not add such constraints.
//
// FINAL DECISION: Return to the EXACT strategy of attempt 9/37, which was the best.
// The only changes:
// 1. Use ConfidenceGatedStep for a trial - this is the most principled change
//    ConfidenceGatedStep uses the model's top token when it's valid, hard-masks otherwise
//    This might reduce the "constrained to wrong schema tokens" problem
//
// Actually, let me reread: ConfidenceGatedStep returns EOS or extends parser validly.
// "avoid using this helper for exact visible spans that must remain fully hard-controlled"
// But our SQL span IS exact - we want hard control. So ConfidenceGatedStep might reduce
// syntax validity.
//
// FINAL ANSWER: Keep exactly the best strategy (GroupBoostedConstrainedStep 6.0, 5-step preamble).
// Only change: improve the guidance string to be more targeted.
// The guidance should emphasize: write EXACTLY what the question asks, don't over-elaborate.
// This might reduce the long wrong answers (death + JOIN ship pattern).
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
// Phase 1 (preamble, UnconstrainedStep loop with !insideConstrainedOut guard):
//   The loop condition includes !insideConstrainedOut. The loop body only runs when
//   !insideConstrainedOut. If "<<" is emitted naturally, we set currentConstrainedOut := []
//   which satisfies parser.IsValidPrefix([]) by method precondition. For any other token,
//   insideConstrainedOut stays false making the implication vacuously true.
//
// Phase 2 (OpenConstrainedSpan):
//   Only reached when !insideConstrainedOut. OpenConstrainedSpan sets
//   currentConstrainedOut := [] and insideConstrainedOut := true.
//   parser.IsValidPrefix([]) holds by the method precondition.
//
// Phase 3 (constrained generation loop):
//   IsCompletePrefix branch: CloseConstrainedSpan sets insideConstrainedOut := false
//   (and currentConstrainedOut := []), making the implication vacuously true.
//   GroupBoostedConstrainedStep returns EOS or a parser-valid token under hard masking.
//   AppendConstrainedToken extends currentConstrainedOut by the valid next token,
//   preserving IsValidPrefix by the parser's forward-validity contract.
//   EOS path + complete close: same as IsCompletePrefix branch.
//   EOS path + not complete: break without modifying span state, invariant holds.
//
// Phase 4 (CloseSpanWithinBudget):
//   Postcondition guarantees insideOut ==> parser.IsValidPrefix(currentOut). Preserved.
//
// progress: |generated| <= |generatedPrefix| + steps
//
// Phase 1: Each UnconstrainedStep: steps += 1. If non-EOS, generated grows by 1.
//   If EOS, break without growing. So |generated| <= |generatedPrefix| + steps after each iteration.
//
// Phase 2: OpenConstrainedSpan: appends exactly 1 "<<", steps += 1.
//   Both sides increment by 1. Invariant preserved.
//
// Phase 3:
//   IsCompletePrefix branch: CloseConstrainedSpan appends at most 1 ">>", steps += 1.
//   Both sides increment by at most 1. Invariant preserved.
//   GroupBoostedConstrainedStep: steps += 1. AppendConstrainedToken appends 1 token (non-EOS).
//   Net: steps += 1, generated += 1. Invariant preserved.
//   EOS + close branch: steps += 2, generated grows by at most 1 (">>" from close).
//   |generated| <= |generatedPrefix| + steps since 1 <= 2.
//   EOS + no close branch: steps += 1 (from GroupBoostedConstrainedStep), generated unchanged.
//
// Phase 4: CloseSpanWithinBudget uses at most closeBudget = maxSteps - steps token-steps.
//   By its postcondition, |generatedOut| <= |generated| + closeBudget = |generated| + (maxSteps - steps).
//   Setting steps := maxSteps: |generatedOut| <= |generatedPrefix| + steps (old) + maxSteps - steps (old)
//   = |generatedPrefix| + maxSteps. Invariant preserved.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Output the SQL query in this format: SQL: <<your query here>>. Use only the exact table names and column names from the provided schema. Answer the question directly with a precise SQL SELECT statement.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;

// Phase 1: Short unconstrained preamble (at most 5 steps) to emit "SQL: " prefix
var preambleBudget: nat := 5;
var preambleSteps: nat := 0;

while steps < maxSteps && preambleSteps < preambleBudget && !insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant 0 <= preambleSteps <= preambleBudget
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  var next := helpers.UnconstrainedStep(lm, prompt, generated);
  steps := steps + 1;
  preambleSteps := preambleSteps + 1;
  if next == eosToken {
    break;
  } else {
    generated := generated + [next];
    if next == "<<" {
      insideConstrainedOut := true;
      currentConstrainedOut := [];
    }
  }
}

// Phase 2: Force "<<" if not already inside constrained span
if !insideConstrainedOut && steps < maxSteps {
  var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
  generated := og;
  insideConstrainedOut := oi;
  currentConstrainedOut := oc;
  steps := steps + 1;
}

// Phase 3: Generate SQL under hard constrained control with group boosting
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
    break;
  } else {
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.GroupBoostedConstrainedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 6.0, eosToken
    );
    steps := steps + 1;
    if next == eosToken {
      // Try to close cleanly if complete
      if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
        var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
          lm, parser, generated, currentConstrainedOut
        );
        generated := closedGenerated;
        insideConstrainedOut := closedInside;
        currentConstrainedOut := closedCurrent;
        steps := steps + 1;
      }
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

// Phase 4: If span still open and budget remains, use CloseSpanWithinBudget
if insideConstrainedOut && steps < maxSteps {
  var closeBudget := maxSteps - steps;
  var cg, ci, cc := helpers.CloseSpanWithinBudget(
    lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
  );
  generated := cg;
  insideConstrainedOut := ci;
  currentConstrainedOut := cc;
  steps := maxSteps;
}

cost := steps;
