// CSD_RATIONALE_BEGIN
// ANALYSIS OF ATTEMPT 19 CATASTROPHIC FAILURE (accuracy 8.2%, syntax 22.4%):
//
// ROOT CAUSE: The attempt introduced CSDHelpers.PrefixToString and CSDHelpers.CountSubstring
// which are NOT in the allowed helper set. These caused runtime errors or exceptions,
// resulting in 3 examples timing out (600s+) and 1 example with no output.
// The `SafePenalizeTokenLogits` call syntax was also invalid in that form.
//
// Also: `helpers.RollbackConstrainedToComplete` is NOT in the allowed helper set.
// The attempt used several disallowed helpers, causing compilation/runtime failures.
//
// RETURN TO BEST RESULT: Attempt 18 gave 49.0% accuracy, 89.8% syntax.
// Attempt 18 used the same structure as attempt 15 (the best).
//
// FAILURE MODES IN ATTEMPT 18 (the best):
// 1. mode_O (3): unterminated spans - EnterObservedConstrainedSpan firing mid-reasoning
// 2. mode_F (2): malformed free-text spans
// 3. mode_I (7): syntax_valid but wrong answers (semantic, not CSD failure)
// 4. mode_K (5): tiny spans (forced-open produces minimal correct-syntax wrong-semantics)
//
// NEW ATTEMPT 20 STRATEGY: Build directly on attempt 18 (the best).
//
// KEY CHANGES FROM ATTEMPT 18:
//
// 1. FOR mode_O (3 unterminated spans):
//    The cause is EnterObservedConstrainedSpan firing at step ~820 when model writes
//    "<<" in mid-reasoning (not as a final answer). The model then tries to write
//    something like "int(form!" where "form!" is invalid → stuck.
//    
//    FIX: Increase observedSpanThreshold so that "<<" only triggers constrained mode
//    very close to the end. Specifically: observedSpanThreshold = prefixBudget.
//    This means: only enter observed mode if the "<<" appears AT OR AFTER step 800.
//    If "<<" appears before step 800, just append it as text.
//    
//    Implementation: check `if steps >= observedSpanThreshold` before calling
//    EnterObservedConstrainedSpan.
//
//    BUT: In attempt 18, the failing examples had "<<" at step ~820 which IS >= 800.
//    So this threshold change alone won't help for these 3 cases.
//
//    BETTER FIX: After entering observed span, if constrained generation hits EOS
//    early (dead end), immediately call CloseSpanWithinBudget with ALL remaining budget
//    instead of the limited nearBudgetThreshold budget.
//    Also: INCREASE nearBudgetThreshold from 30 to 50, so we switch to CloseSpanWithinBudget
//    earlier and give it more steps.
//
// 2. FOR mode_F (2 malformed free-text spans):
//    These are cases where the model writes "<<int(n0 * (1 + r) ** d)>>" in free text.
//    The "**" operator makes it invalid.
//    FIX: Add guidance that explicitly says to use "*" not "**".
//    We already do this in attempt 18, but the model ignores it.
//    ADDITIONAL FIX: Use `SafePenalizeTokenLogits` before each UnconstrainedStep
//    to discourage "**" and "^" tokens.
//    BUT this costs 0 per call and we can call it before each UnconstrainedStep.
//    The issue is that logit edits are wiped at next GenerateLogits call.
//    SafePenalizeTokenLogits requires calling BEFORE the GenerateLogits call that
//    feeds into UnconstrainedStep. UnconstrainedStep calls GenerateLogits internally.
//    So we can't inject penalties between GenerateLogits and ChooseNextToken.
//    THEREFORE: SafePenalizeTokenLogits before UnconstrainedStep has NO effect
//    because UnconstrainedStep regenerates logits internally.
//    
//    ALTERNATIVE: The SafePenalizedConstrainedStep or similar inside-span helpers
//    can apply penalties. But for free text, we can't inject penalties.
//    
//    BEST APPROACH: Just improve the guidance text to be very explicit.
//    Also: ensure force-open happens earlier (prefixBudget=700) so model has LESS
//    unconstrained text to write before the forced constrained span.
//    This reduces the chance of model writing "**" in free text.
//
// 3. FOR mode_I (7 syntax-valid but wrong):
//    These are semantic failures - the model computes the wrong formula.
//    Not directly fixable by CSD strategy, but we can try:
//    - Better guidance that emphasizes the reasoning pattern
//    - The guidance already says "solve step by step"
//    - These are genuinely hard problems
//    - Cannot fix without grounding/oracle access
//
// 4. FOR mode_K (5 tiny spans):
//    The forced-open constrained span produces "<<int(n)>>" or similar.
//    This happens when model doesn't have enough context at force-open point.
//    FIX: Force-open LATER (after more reasoning). Keep prefixBudget=800.
//    ALSO: Increase maxSpanTokens to allow more constrained generation.
//
// DIAGNOSTIC DATA FROM ATTEMPT 18:
// - Generated tokens/example: avg 820 → examples are using nearly full budget
// - examples_without_activity: 36/49 → model writes spans naturally for most
// - examples_with_activity: 13/49 → 13 cases where force-open triggered
// - correct_without_activity: 23/49 (64% when model uses free text)
// - correct_with_activity: 1/49 (8% when force-open triggered)
// - The free-text approach (36 examples) is much better than force-open (13 examples)
//
// ATTEMPT 20 REFINED PLAN:
//
// Change 1: observedSpanThreshold = prefixBudget (800)
//   Only enter observed constrained span if the "<<" appears at steps >= 800.
//   Before step 800, "<<" is just appended as text.
//   This prevents early observed span entry (which was causing confusion in old attempts).
//   Wait: the 3 mode_O failures had "<<" at ~820 steps (>= 800), so this doesn't help them.
//   But it prevents NEW failures from early "<<" triggering.
//
// Change 2: nearBudgetThreshold = 50 (was 30 in attempt 18)
//   Switch to CloseSpanWithinBudget when remainingBudget <= 50.
//   This gives 50 steps for forced closure vs 30 before.
//   More budget for CloseSpanWithinBudget to find a valid completion.
//
// Change 3: When EOS in constrained phase → use ALL remaining budget for CloseSpanWithinBudget
//   When AdaptiveConstrained returns EOS, use rem = maxSteps - steps (not just nearBudgetThreshold).
//   This maximizes chances of closure.
//
// Change 4: maxSpanTokens = 25 (was 22 in attempt 18)
//   Allow a few more tokens in the constrained span before forcing close.
//   This helps generate longer valid expressions.
//
// Change 5: Guidance update
//   Make guidance more direct about not using "**" or "^" and
//   using only the exact variable names from the problem (no curly braces, no markdown).
//
// SUMMARY OF CHANGES FROM ATTEMPT 18:
// - observedSpanThreshold: added (only trigger observed entry at steps >= prefixBudget)
// - nearBudgetThreshold: 30 → 50
// - maxSpanTokens: 22 → 25
// - EOS handler: use all remaining budget instead of limited budget
// - Guidance: minor refinement
//
// These are conservative, targeted changes to the best-performing strategy.
// No new helper calls, no structural changes.
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// Invariant 1: parser_validity
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
// Branch A (UnconstrainedStep, next != "<<" or steps < observedSpanThreshold):
//   insideConstrainedOut unchanged (false). Implication vacuously true.
//
// Branch B (UnconstrainedStep, next == "<<", steps >= observedSpanThreshold):
//   generated += ["<<"]; EnterObservedConstrainedSpan sets insideConstrainedOut := true,
//   currentConstrainedOut := []. parser.IsValidPrefix([]) holds by precondition. Preserved.
//
// Branch C (OpenConstrainedSpan force-open, steps >= prefixBudget):
//   OpenConstrainedSpan sets insideConstrainedOut := true, currentConstrainedOut := [].
//   parser.IsValidPrefix([]) by precondition. Preserved.
//
// Branch D (CloseSpanIfComplete, closed=true):
//   insideConstrainedOut := false. Implication vacuously true.
//
// Branch E (CloseSpanIfComplete, closed=false, AdaptiveConstrained, next != EOS):
//   AdaptiveConstrainedStepWithPenalties returns parser-valid token by contract.
//   AppendConstrainedToken: currentConstrainedOut := currentConstrainedOut + [next].
//   IsValidPrefix(currentConstrainedOut + [next]) holds. Preserved.
//
// Branch F (AdaptiveConstrained returns EOS, or nearBudget, or maxSpanTokens):
//   CloseSpanWithinBudget postcondition guarantees:
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut). Preserved.
//
// Invariant 2: progress
//   |generated| <= |generatedPrefix| + steps
//
// Branch A (UnconstrainedStep + append, steps += 1):
//   |generated| grows by 1 (one token). steps grows by 1. Preserved.
//
// Branch B (EnterObservedConstrainedSpan after "<<" observed, steps already += 1):
//   "<<" was appended (steps += 1, generated grew by 1) before calling Enter.
//   EnterObservedConstrainedSpan: 0 cost, no extra append. Preserved.
//
// Branch C (OpenConstrainedSpan, steps += 1):
//   Appends "<<" to generated (1 token). steps += 1. |generated| = old+1, steps = old+1. Preserved.
//
// Branch D (CloseSpanIfComplete closed, steps += 1):
//   CloseConstrainedSpan appends ">>" (1 token). |generated| += 1. steps += 1. Preserved.
//
// Branch E (CloseSpanIfComplete not closed, steps += 1, then AppendConstrainedToken):
//   CloseSpanIfComplete: steps += 1, no visible change.
//   AppendConstrainedToken: |generated| += 1, steps unchanged.
//   Net: |generated| += 1, steps += 1. Preserved.
//
// Branch F (CloseSpanWithinBudget with rem = maxSteps - steps):
//   rem <= maxSteps - steps (by definition).
//   |generated| grows by at most rem; steps grows by rem.
//   |generated| <= |generatedPrefix| + steps + rem = |generatedPrefix| + maxSteps. Preserved.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve step by step. Use the variable names exactly as they appear in the problem WITHOUT curly braces. End with exactly one <<int(formula)>> for the final answer. Use only +, -, *, /, //, % operators. Never use ** or ^ for exponentiation - use * instead. Examples: <<int(n * p)>>, <<int(a - b * c)>>, <<int(total // count)>>.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;
var prefixBudget: nat := if maxSteps > 200 then maxSteps - 100 else (if maxSteps > 100 then maxSteps - 30 else maxSteps / 2);
var observedSpanThreshold: nat := prefixBudget;
var penaltyTokens: seq<Token> := ["{", "}", "**", "^", "\\"];
var spanTokens: nat := 0;
var maxSpanTokens: nat := 25;
var nearBudgetThreshold: nat := if maxSteps > 60 then 50 else maxSteps / 4;

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
    var remainingBudget := maxSteps - steps;
    if steps >= prefixBudget && remainingBudget > 5 {
      // Force open a constrained span for the final answer
      var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
      generated := og;
      insideConstrainedOut := oi;
      currentConstrainedOut := oc;
      steps := steps + 1;
      spanTokens := 0;
    } else {
      var next := helpers.UnconstrainedStep(lm, prompt, generated);
      steps := steps + 1;
      if next == eosToken {
        break;
      } else if next == "<<" && steps >= observedSpanThreshold {
        // Near end of prefix budget: enter observed constrained span
        generated := generated + [next];
        var eg, ei, ec := helpers.EnterObservedConstrainedSpan(lm, generated);
        generated := eg;
        insideConstrainedOut := ei;
        currentConstrainedOut := ec;
        spanTokens := 0;
      } else {
        generated := generated + [next];
      }
    }
  } else {
    // Inside constrained span
    var remainingBudget := maxSteps - steps;

    // If near budget end or span too long: force close with all remaining budget
    if remainingBudget <= nearBudgetThreshold || spanTokens >= maxSpanTokens {
      if remainingBudget > 0 {
        var sg, si, sc := helpers.CloseSpanWithinBudget(
          lm, parser, prompt, generated, currentConstrainedOut, eosToken, remainingBudget
        );
        generated := sg;
        insideConstrainedOut := si;
        currentConstrainedOut := sc;
        steps := steps + remainingBudget;
      }
      break;
    }

    // Try to close if complete (costs 1 step)
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    steps := steps + 1;
    if closed {
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      break;
    } else {
      // Generate next constrained token
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.AdaptiveConstrainedStepWithPenalties(
        lm, parser, constrainedPrompt, currentConstrainedOut,
        validTokenGroups, 2.0,
        penaltyTokens, 8.0,
        6, eosToken
      );
      if next == eosToken {
        // EOS inside span: use all remaining budget for forced close
        var rem := maxSteps - steps;
        if rem > 0 {
          var sg2, si2, sc2 := helpers.CloseSpanWithinBudget(
            lm, parser, prompt, generated, currentConstrainedOut, eosToken, rem
          );
          generated := sg2;
          insideConstrainedOut := si2;
          currentConstrainedOut := sc2;
          steps := steps + rem;
        }
        break;
      } else {
        var ag, ai, ac := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := ag;
        insideConstrainedOut := ai;
        currentConstrainedOut := ac;
        spanTokens := spanTokens + 1;
      }
    }
  }
}

cost := steps;
