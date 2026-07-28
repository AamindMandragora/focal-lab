// CSD_RATIONALE_BEGIN
// Diagnosis of all previous attempts:
// - The "!!!" pattern appears because SafeRepetitionPenaltyStep is generating "!" tokens
//   in the constrained span. This suggests the SQL grammar parser is accepting "!" tokens.
// - "valid_nonfinal_spans_only 300/300" and "final_answer_span 0/300" means the
//   constrained span content is never the FINAL answer. The evaluator extracts from
//   the "last_visible_span" but there are 0 visible spans completing cleanly.
// - "Examples with visible `<<`: 1/300" - OpenConstrainedSpan IS being called but the
//   evaluator doesn't see visible <<. This is confusing. Wait - the diagnostic says
//   "Examples with visible `<<`: 1/300" but "valid_nonfinal_spans_only 300/300".
//   This means the constrained activity is happening internally (hidden) but the visible
//   << >> delimiters are not in the output text.
// - "unmatched visible close: 92/300" - 92 examples have ">>" in output without "<<"!
//   This is because the constrained grammar itself emits ">>" as a SQL token (like ">>").
//   Actually no - ">>" is the closing delimiter. The SQL content has ">>" somehow.
//
// KEY INSIGHT: Looking at the rollout "SELECT id FROM high>>":
// The ">>" appears INSIDE what should be the SQL, cutting it short.
// The SQL grammar parser is treating ">>" as a valid SQL token (like a bitshift operator
// or part of some schema), which causes CloseConstrainedSpan to fire early!
//
// Another KEY INSIGHT: The model output is "SELECT id FROM high>>" - this has ">>" 
// visible but NO "<<". This means: the constrained span was opened INTERNALLY (hidden mode),
// the model generated some SQL, then ">>" appeared in the SQL grammar output itself,
// which triggered span close - but the << was never visible!
//
// CRITICAL FIX: The issue is that the strategy opens a constrained span but the
// "<<" is internal/hidden. The evaluator needs VISIBLE "<<" followed by VISIBLE ">>".
// We need to use the explicit approach where WE emit "SQL: <<" before generating SQL.
//
// Strategy for attempt 7:
// 1. Very short unconstrained phase (4 steps) to get "SQL: " 
// 2. Force OpenConstrainedSpan explicitly (this appends visible "<<")
// 3. Use RepetitionPenaltyStep for constrained SQL generation
//    - RepetitionPenaltyStep discourages repeating tokens (prevents "!!!" loops)
// 4. Reserve 200 steps for CloseSpanWithinBudget
//
// The RepetitionPenaltyStep IS in the allowed helper list and takes `generated`
// as context for penalization - this should prevent the "!!!..." repetition.
//
// Key difference from attempt 6: RepetitionPenaltyStep vs SafeRepetitionPenaltyStep
// Both are in the allowed list. Let's use RepetitionPenaltyStep which requires
// vocabulary membership. Instead, use SafeSoftConstrainedStep which combines
// soft grammar preference with hard parser fallback and is robust.
//
// Actually the real fix: the output shows SQL is truncated early ("SELECT id FROM high>>")
// This means CloseSpanWithinBudget or CloseSpanIfComplete is firing too early because
// the constrained grammar considers "SELECT id FROM high" as a "complete" prefix.
//
// For the "!!!..." examples: SafeRepetitionPenaltyStep is generating "!" tokens.
// The SQL parser must be accepting "!" as valid. We need to avoid this.
//
// REVISED PLAN: Use ConstrainedGeneration which loops ConstrainedStep until parser
// completeness, EOS, or budget. This is the cleanest approach - it generates a
// complete SQL from scratch in constrained mode. Then wrap it in visible << >>.
//
// After ConstrainedGeneration returns the SQL, we wrap it: emit "<<" + SQL + ">>"
// But we can't easily do that post-hoc. Instead:
// 1. Short unconstrained (4 steps) for "SQL: "
// 2. OpenConstrainedSpan (emits visible "<<")
// 3. Use the main constrained loop with ConstrainedStep + CloseSpanWithinBudget
//    but DON'T call CloseSpanIfComplete during generation (let it run to budget)
//    Instead, only close at the very end with CloseSpanWithinBudget.
//
// This prevents premature closing when the grammar accepts partial SQL as "complete".
//
// FINAL PLAN:
// - Phase 1: 4 unconstrained steps for "SQL: "
// - Phase 2: OpenConstrainedSpan for visible "<<"
// - Phase 3: Generate SQL using ConstrainedStep loop, NO CloseSpanIfComplete
//   until we've generated at least some meaningful content (min 20 tokens).
//   Then start checking for completion.
// - Phase 4: CloseSpanWithinBudget with large budget.
//
// Use SafeSoftConstrainedStep for generation - it has soft grammar preference
// with hard fallback, which should produce more natural SQL.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   1. Initialization: insideConstrainedOut = insideConstrained,
//      currentConstrainedOut = currentConstrained (from preconditions).
//   2. Phase 1 (unconstrained, at most 4 steps): Each UnconstrainedStep returns
//      one token. If next == "<<", we set insideConstrainedOut = true and
//      currentConstrainedOut = [], which satisfies parser.IsValidPrefix([]).
//      Otherwise insideConstrainedOut stays false and currentConstrainedOut stays [].
//   3. Phase 2 (OpenConstrainedSpan): postcondition guarantees insideConstrainedOut = true,
//      currentConstrainedOut = [], and parser.IsValidPrefix([]) holds.
//   4. Phase 3 (constrained generation loop):
//      - SafeSoftConstrainedStep returns a token that is EOS or keeps
//        currentConstrainedOut + [next] parser-valid.
//      - AppendConstrainedToken preserves parser.IsValidPrefix(currentConstrainedOut).
//      - After min content (minContent steps), CloseSpanIfComplete either closes
//        (implication vacuous) or leaves state unchanged (no-op, valid).
//      - Loop invariant: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut).
//   5. Phase 4 (CloseSpanWithinBudget): postcondition guarantees:
//      insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//      !insideConstrainedOut ==> currentConstrainedOut == []
//
// progress (|generated| <= |generatedPrefix| + steps):
//   Phase 1: Each step increments steps by 1 and appends at most 1 token.
//   Phase 2 (OpenConstrainedSpan): costs 1 step, appends "<<" (1 token).
//   Phase 3: Each iteration of the inner loop: SafeSoftConstrainedStep costs 1 step;
//     AppendConstrainedToken appends 1 token. CloseSpanIfComplete costs 1 step and
//     appends at most 1 token (">>"). So per loop iteration steps increases by 1 or 2,
//     |generated| increases by at most 1 per token appended. Bound holds.
//   Phase 4 (CloseSpanWithinBudget with closeBudget = maxSteps - steps):
//     |generatedOut| <= |generated| + closeBudget <= |generatedPrefix| + maxSteps.
//     We set steps := maxSteps. Bound holds.
//   cost = steps <= maxSteps. maxSteps > 0 implies Phase 2 runs (cost >= 1 > 0).
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Generate a SQL query. Output format must be: SQL: <<SELECT ...>> with the complete SQL query inside the << >> delimiters. Use exact table and column names from the schema. Complete the full WHERE clause, JOIN conditions, and ORDER BY as needed.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;

// Phase 1: Short unconstrained prefix (at most 4 steps) to emit "SQL: "
var prefixBudget: nat := 4;
if prefixBudget > maxSteps {
  prefixBudget := maxSteps;
}

while steps < prefixBudget && !insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases prefixBudget - steps
{
  var next := helpers.UnconstrainedStep(lm, prompt, generated);
  steps := steps + 1;
  if next == eosToken {
    cost := steps;
    return;
  }
  generated := generated + [next];
  if next == "<<" {
    insideConstrainedOut := true;
    currentConstrainedOut := [];
  }
}

// Phase 2: Force open constrained span if not already inside one
if !insideConstrainedOut && steps < maxSteps {
  var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
  generated := og;
  insideConstrainedOut := oi;
  currentConstrainedOut := oc;
  steps := steps + 1;
}

// Phase 3: Generate SQL inside constrained span
// Use a minimum content threshold before allowing CloseSpanIfComplete
// This prevents premature closure on partial SQL
var minContentSteps: nat := 15;
var contentSteps: nat := 0;
// Reserve substantial budget for CloseSpanWithinBudget
var reserveForClose: nat := 250;

while steps + reserveForClose < maxSteps && insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  // Only try to close after generating minimum content
  if contentSteps >= minContentSteps {
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    steps := steps + 1;
    if closed {
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      // Exit loop - span is closed
    } else {
      // Generate next token with soft constraint + hard fallback
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next, usedFallback := helpers.SafeSoftConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, 3.0, eosToken);
      steps := steps + 1;
      if next == eosToken {
        break;
      }
      var ag, ai, ac := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := ag;
      insideConstrainedOut := ai;
      currentConstrainedOut := ac;
      contentSteps := contentSteps + 1;
    }
  } else {
    // Generate next token without trying to close yet
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next, usedFallback := helpers.SafeSoftConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, 3.0, eosToken);
    steps := steps + 1;
    if next == eosToken {
      break;
    }
    var ag, ai, ac := helpers.AppendConstrainedToken(
      lm, parser, generated, currentConstrainedOut, next
    );
    generated := ag;
    insideConstrainedOut := ai;
    currentConstrainedOut := ac;
    contentSteps := contentSteps + 1;
  }
}

// Phase 4: Close the span if still open using remaining budget
if insideConstrainedOut && steps < maxSteps {
  var closeBudget: nat := maxSteps - steps;
  var cg, ci, cc := helpers.CloseSpanWithinBudget(
    lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
  );
  generated := cg;
  insideConstrainedOut := ci;
  currentConstrainedOut := cc;
  steps := maxSteps;
}

cost := steps;
