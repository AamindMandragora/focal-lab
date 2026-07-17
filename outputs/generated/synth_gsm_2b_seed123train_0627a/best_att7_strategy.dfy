// CSD_RATIONALE_BEGIN
// Math word problem CSD with visible << >> delimiters.
//
// Analysis of failure pattern:
// - Best attempt (2): 42.9% accuracy, 79.6% syntax (10 unclosed spans)
// - Current attempt (6): 38.8% accuracy, 63.3% syntax (18 unclosed spans)
// - The token-by-token approach in attempt 6 INCREASED unclosed spans vs attempt 2.
//
// Why attempt 6 was worse than attempt 2:
// - Attempt 6 used CloseSpanIfComplete + ConstrainedStep in a loop.
// - CloseSpanIfComplete costs 1 step EVEN WHEN it returns closed=false (no-op).
// - ConstrainedStep costs another 1 step per token.
// - So each token inside the span costs 2 steps! This burns the budget faster.
// - With the same freeChunkBudget (maxSteps - 60), there's far less budget
//   for the constrained generation.
//
// Why attempt 2 is better:
// - Uses CloseSpanWithinBudget which is a single budget-efficient call.
// - CloseSpanWithinBudget internally tracks the last complete state and
//   always closes if any complete state was found.
//
// Why attempt 2 still has 10 unclosed spans:
// The feedback says all 18 unclosed spans in attempt 6 are "final_span_unclosed"
// type, where "<<" was emitted at the END and then stopped before any content.
// The output tails show patterns like:
//   "...The final symbolic expression is `{g} - {n_1} - 3 * {n_2}`.\n\n<<n1 * 3 + g"
//   "...{n} * (1 + {k_2} + {k_3} + 0.5*{k_3} + 0.125*{k_3}*{n}\nTotal = {n} * ..."
//
// KEY INSIGHT: The model generates very long reasoning (~168 tokens average before "<<")
// BUT some examples have the model write "<<" and then enter the span with 
// a small remaining budget. If freeChunkBudget = maxSteps - 60 = 840, and the 
// model generates "<<" at token 835, then closeBudget = 900 - 835 - 1 = 64.
// But the third example shows "<<relative_age years!!!!..." - the model wrote
// "relative_age" as content but the grammar likely doesn't accept identifier_name
// as a valid token (it may need "relative_age" without underscores or with different
// tokenization).
//
// The real issue: when "<<" appears NEAR THE END of a long reasoning chain, the
// content after "<<" may be partially valid text that the model wrote INLINE
// (like "<<n1 * 3 + g" which IS a valid arithmetic expression start), but 
// CloseSpanWithinBudget needs to extend it to completion and close.
//
// HYPOTHESIS: The freeChunkBudget is too high. When the model generates "<<" at
// token 800 (out of 840 budget), closeBudget = 900 - 800 = 100. That should be
// enough. But if it's at token 839 (the last allowed token before the force),
// closeBudget = 900 - 840 = 60. Still enough for a short expression.
//
// Let me look more carefully at the feedback: "final_span_unclosed: 18 example(s) —
// the generation emitted `<<` at the end and then stopped (EOS or dead-end) before
// producing any span content or `>>>`."
//
// "stopped (EOS or dead-end)" - the constrained generation STOPPED early!
// This means the parser is hitting a dead-end immediately after "<<". 
// The arithmetic expression grammar expects certain tokens after "<<" but
// the model wants to emit tokens that are not in the grammar's valid set.
//
// OR: CloseSpanWithinBudget couldn't find any complete state because:
// The model tries to emit variable names with braces "{n1}" or subscripts which
// are not valid in the arithmetic expression grammar. The grammar only allows
// simple identifiers (a-z, _, digits). So "{n1}" would fail immediately.
//
// Actually re-reading the failing output:
// "<<n1 * 3 + g" - this IS valid arithmetic (n1, 3, g are simple identifiers).
// So the span content is valid but UNCLOSED. This means CloseSpanWithinBudget 
// ran but didn't close it. With budget ~60 after this, that's plenty.
//
// CRITICAL: The output shows "<<n1 * 3 + g" as the tail, meaning the generation
// STOPPED at "g". After generating valid tokens, the model hit its generation
// limit INSIDE CloseSpanWithinBudget without emitting ">>".
//
// Wait - the feedback says "steps budget ran out". The 900 step budget was used
// up without ">>".
//
// Actually looking at this: "avg 168.04 tokens before first visible open".
// Average is 168, but SOME examples have much more. If an example uses 850+ tokens
// of free generation before "<<", then only 49 tokens remain for the constrained part.
// If freeChunkBudget = maxSteps - 60 = 840, and stepsUsed = 840, then steps = 840.
// Force open: steps = 841 (if model never emitted "<<").
// closeBudget = 900 - 841 = 59. That should be fine.
//
// But if the model DID emit "<<" at step 839 (the last step of UnconstrainedChunk),
// then closeBudget = 900 - 839 = 61. Still fine.
//
// WAIT: Looking at "Tokens before first visible open: avg 168.04, median 148.00".
// These are TOKENS GENERATED, not steps. Each unconstrained step = 1 token.
// So the model typically writes "<<" after ~168 tokens of free text.
// With freeChunkBudget = 840, most models emit "<<" well within budget,
// leaving ~732 steps for the constrained part. That's PLENTY.
//
// So why 18 unclosed spans? 
//
// ANOTHER CLUE: "Constrained helper call fraction: 0.97" - almost ALL calls are
// constrained. But "ConstrainedStep=839, AppendConstrainedToken=823".
// With 49 examples, that's 17 constrained steps per example on average.
// But "CloseConstrainedSpan=31" = 31/49 examples closed. For the 18 that didn't,
// there were LOTS of constrained steps (839/49 ≈ 17 per example, total).
//
// Wait, I'm looking at attempt 6's numbers (the token-by-token approach).
// For attempt 2 (best), the helper calls would show UnconstrainedChunk + 
// CloseSpanWithinBudget only. Those metrics are from attempt 6.
//
// Let me refocus on attempt 2's failures. Attempt 2 had 79.6% syntax = 39/49 closed.
// 10 unclosed. The output tails show the model generating "<<" near the end.
//
// PLAN TO FIX:
// 1. Keep the same structure as attempt 2 (best result).
// 2. Reduce freeChunkBudget more aggressively (e.g., maxSteps - 100 or maxSteps/2)
//    to ensure MORE budget for the constrained span.
// 3. Use a hybrid: UnconstrainedChunk to let model write its natural answer
//    INCLUDING "<<", then use CloseSpanWithinBudget with a generous budget.
//
// Actually the real problem might be simpler: the arithmetic expression grammar
// is quite expressive and allows very long expressions. After "<<", the model
// generates a long expression chain that fills the remaining budget without 
// ever reaching a "complete" state (because the model keeps extending).
//
// CRITICAL OBSERVATION: CloseSpanWithinBudget spec says:
// "If no completable state is reached within budget, the span is left open."
// This is the failure mode! If the parser never reports IsCompletePrefix for
// ANY prefix within the budget, CloseSpanWithinBudget leaves the span open.
//
// When would IsCompletePrefix never be true? If the arithmetic expression grammar
// requires something more than just tokens to be "complete". For example, if
// the grammar is defined as a sequence that expects a closing token (like ">>")
// as part of completeness, then the grammar would NEVER be complete inside the
// span.
//
// OR: The grammar only accepts certain token sequences as complete expressions.
// Simple cases like "42" or "n1 * 2" would be complete. But if the model 
// generates a partial expression like "n * p1 * p2 * frac /" that's NOT complete
// (ends with an operator), and CloseSpanWithinBudget generates forward but the
// model keeps adding operators...
//
// FINAL DIAGNOSIS: The issue is timing. Some examples generate very long 
// reasoning (400+ tokens) before "<<", consuming budget, and then the arithmetic
// expression inside the span either (a) gets cut off by budget, or (b) the model
// generates invalid content (variable names with special chars) that the parser
// rejects immediately.
//
// STRATEGY: Use a MUCH shorter free text budget. Only let the model write ~200 
// tokens of reasoning. If "<<" hasn't appeared by then, force it open.
// This ensures at least 700 tokens for the constrained span, which is more than
// enough to close any arithmetic expression.
//
// Additionally: use task guidance that strongly encourages the model to write
// "<<" EARLY, not at the end of a long chain of reasoning.
//
// REVISED PLAN (attempt 7):
// Phase 1: UnconstrainedChunk with budget = min(maxSteps - 200, 300)
//   - Let model write ~300 tokens of reasoning, stopping on "<<".
//   - Reserve at least 200 tokens for the constrained span.
// Phase 2: If model never emitted "<<", force open.
// Phase 3: CloseSpanWithinBudget with remaining budget (generous).
//
// This gives the constrained span at least 200 tokens, usually much more.
// Even for edge cases where the model uses all 300 free tokens, closeBudget = 600.
//
// Task guidance: Make it shorter and more direct to get the model to emit "<<" quickly.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity invariant:
//   - Phase 1 (UnconstrainedChunk): When stoppedOnOpenSpan is true, generated
//     already ends with "<<" and EnterObservedConstrainedSpan sets
//     insideConstrainedOut := true, currentConstrainedOut := [] (which satisfies
//     parser.IsValidPrefix([]) by precondition). When stoppedOnEos, we return.
//     Otherwise insideConstrainedOut remains false (invariant vacuously true).
//   - Force-open branch (OpenConstrainedSpan): sets insideConstrainedOut := true,
//     currentConstrainedOut := [] (valid by precondition). Invariant preserved.
//   - Phase 3 (CloseSpanWithinBudget): postcondition guarantees either
//     (!insideConstrainedOut && currentConstrainedOut == []) or
//     (insideConstrainedOut && parser.IsValidPrefix(currentConstrainedOut)).
//     Both branches preserve the invariant.
//
// progress invariant (|generated| <= |generatedPrefix| + steps):
//   - Phase 1: UnconstrainedChunk costs stepsUsed and grows generated by at most
//     stepsUsed tokens (EOS not appended). steps := steps + stepsUsed preserves bound.
//   - Force-open: OpenConstrainedSpan costs 1 step, appends "<<" (1 token).
//     steps := steps + 1 preserves the bound.
//   - Phase 3: CloseSpanWithinBudget with closeBudget = maxSteps - steps guarantees
//     |generatedOut| <= |generated| + closeBudget <= |generatedPrefix| + maxSteps.
//     Setting steps := maxSteps preserves cost <= maxSteps.
//   - Final: cost := steps <= maxSteps. Progress (cost > 0 if maxSteps > 0) holds
//     because Phase 1 always takes at least 1 step, or if maxSteps = 0 the loop
//     body is skipped and cost = 0 satisfying the vacuous case.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve step by step. Write the final answer as <<expression>> using only variable names and operators +, -, *, /. Example: <<n * price + tax>>. Write << immediately before your final answer.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;

// Phase 1: Generate unconstrained text, stopping on "<<" or EOS.
// Use a moderate cap: allow up to 400 tokens of free reasoning but reserve
// at least 500 tokens for constrained span generation and closure.
var freeChunkBudget: nat := if maxSteps > 500 then maxSteps - 500 else (if maxSteps > 1 then maxSteps / 2 else maxSteps);

if !insideConstrainedOut && steps < freeChunkBudget {
  var chunkBudget: nat := freeChunkBudget - steps;
  var cg, stoppedOnOpenSpan, stoppedOnEos, stepsUsed :=
    helpers.UnconstrainedChunk(lm, prompt, generated, chunkBudget, "<<", eosToken);
  generated := cg;
  steps := steps + stepsUsed;
  if stoppedOnEos {
    cost := steps;
    return;
  }
  if stoppedOnOpenSpan {
    generated, insideConstrainedOut, currentConstrainedOut :=
      helpers.EnterObservedConstrainedSpan(lm, generated);
  }
}

// Phase 2: If still outside a constrained span (model never emitted "<<"),
// force open one explicitly.
if !insideConstrainedOut && steps < maxSteps {
  generated, insideConstrainedOut, currentConstrainedOut :=
    helpers.OpenConstrainedSpan(lm, generated);
  steps := steps + 1;
}

// Phase 3: Close the constrained span within the generous remaining budget.
// With freeChunkBudget = maxSteps - 500, we have at least 500 steps here.
// CloseSpanWithinBudget will track the longest complete state and close there.
if insideConstrainedOut && steps < maxSteps {
  var closeBudget: nat := maxSteps - steps;
  generated, insideConstrainedOut, currentConstrainedOut :=
    helpers.CloseSpanWithinBudget(
      lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget);
  steps := maxSteps;
}

cost := steps;

