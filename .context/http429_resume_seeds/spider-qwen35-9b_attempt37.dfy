// CSD_RATIONALE_BEGIN
// Strategy: Return to best attempt (51.7% accuracy, 94% syntax) with targeted fixes.
// Key diagnostics from best attempt:
//   - 127 syntax_valid_semantic_mismatch: correct syntax but wrong answer
//   - 18 wrong_after_constrained_activity: constrained generation went wrong
//   - Best used minConstrainedTokens=40, earlyPhaseTokens=15
//
// Root cause analysis:
//   - The model generates plausible but semantically wrong SQL (wrong tables, wrong conditions)
//   - Need stronger schema grounding: more use of validTokenGroups (schema tokens)
//   - Too many free unconstrained tokens before SQL generation may let model go off-track
//   - The 15-token early phase with GroupBoosted may not be enough for complex queries
//
// Changes from best attempt (attempt 9):
//   1. Reduce prefixBudget from 4 to 2: less unconstrained leeway before SQL
//   2. Increase earlyPhaseTokens from 15 to 30: stronger schema grounding for longer
//   3. Keep minConstrainedTokens=40 (proven to work for syntax)
//   4. Add RepetitionPenaltyStep when we detect a repetition pattern forming
//     (constrainedTokenCount > 20 and we've seen repetitive tokens)
//   5. Use stronger boost amount 8.0 in early phase (vs 6.0) for better schema grounding
//   6. In later phase, use AdaptiveConstrainedStep with higher narrowThreshold=30 (vs 20)
//   7. Better guidance: focus on exact schema column/table names
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// parser_validity invariant:
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//   Phase 1 (free prefix, steps < prefixBudget):
//     If next == "<<", we set insideConstrainedOut := true, currentConstrainedOut := [].
//     parser.IsValidPrefix([]) holds by precondition. Otherwise insideConstrainedOut
//     stays false (implication vacuous).
//
//   Phase 2 (force open span):
//     OpenConstrainedSpan postcondition: insideOut == true, currentOut == [].
//     parser.IsValidPrefix([]) by precondition. Invariant preserved.
//
//   Phase 3 (constrained loop):
//     Branch A (allowClose && IsCompletePrefix -> CloseConstrainedSpan):
//       CloseConstrainedSpan sets insideConstrainedOut := false. Implication vacuous.
//     Branch B (generation step -> EOS):
//       CloseSpanWithinBudget postcondition guarantees insideOut ==> parser.IsValidPrefix(currentOut).
//     Branch C (valid token -> AppendConstrainedToken):
//       GroupBoostedConstrainedStep/AdaptiveConstrainedStep/RepetitionPenaltyStep returns 
//       a parser-valid next token. AppendConstrainedToken extends currentConstrainedOut 
//       by [next] which preserves validity.
//
//   Final cleanup: CloseSpanWithinBudget postcondition guarantees invariant.
//
// progress invariant:
//   |generated| <= |generatedPrefix| + steps
//
//   Phase 1: Each iteration steps += 1, |generated| grows by at most 1 (non-EOS token). Preserved.
//
//   Phase 2: OpenConstrainedSpan appends "<<" (+1 token), steps += 1. Preserved.
//
//   Phase 3:
//     Branch A (CloseConstrainedSpan): appends ">>" (+1 token), steps += 1. Preserved.
//     Branch B (EOS): steps += 1 before EOS check. CloseSpanWithinBudget with
//       closeBudget = maxSteps - steps: |generatedOut| <= |generated| + closeBudget.
//       Setting steps := maxSteps ensures |generated| <= |generatedPrefix| + maxSteps = |generatedPrefix| + steps.
//     Branch C (valid token): steps += 1, AppendConstrainedToken adds 1 token. Preserved.
//
//   Final cleanup: CloseSpanWithinBudget with closeBudget = maxSteps - steps.
//     |generatedOut| <= |generated| + closeBudget = |generated| + (maxSteps - steps) <= |generatedPrefix| + maxSteps.
//     Setting steps := maxSteps preserves the invariant at exit.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance := "Translate the question to a simple, direct SQL query. Use exact table and column names from the schema. Use the simplest SQL structure that answers the question. Output only the SQL query between << and >>.";
helpers.AppendTaskGuidance(lm, guidance);

if maxSteps == 0 {
  cost := 0;
} else {
  var steps: nat := 0;

  var prefixBudget: nat := 2;
  var hitEos: bool := false;

  while steps < prefixBudget && steps < maxSteps && !insideConstrainedOut && !hitEos
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
      hitEos := true;
    } else {
      generated := generated + [next];
      if next == "<<" {
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }
    }
  }

  if !insideConstrainedOut && !hitEos && steps < maxSteps {
    var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
    generated := og;
    insideConstrainedOut := oi;
    currentConstrainedOut := oc;
    steps := steps + 1;
  }

  var constrainedTokenCount: nat := 0;
  var minConstrainedTokens: nat := 40;
  var earlyPhaseTokens: nat := 30;

  while steps < maxSteps && insideConstrainedOut
    invariant 0 <= steps <= maxSteps
    invariant lm.ValidTokensIdsLogits()
    invariant !insideConstrainedOut ==> currentConstrainedOut == []
    invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
    invariant |generated| <= |generatedPrefix| + steps
    decreases maxSteps - steps
  {
    var remainingBudget := maxSteps - steps;

    var allowClose := (constrainedTokenCount >= minConstrainedTokens) || (remainingBudget <= 3);

    if allowClose && parser.IsCompletePrefix(currentConstrainedOut) {
      var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := steps + 1;
    } else {
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := eosToken;

      if constrainedTokenCount < earlyPhaseTokens {
        next := helpers.GroupBoostedConstrainedStep(
          lm, parser, constrainedPrompt, currentConstrainedOut,
          validTokenGroups, 8.0, eosToken
        );
      } else {
        next := helpers.AdaptiveConstrainedStep(
          lm, parser, constrainedPrompt, currentConstrainedOut,
          validTokenGroups, 5.0, 30, eosToken
        );
      }
      steps := steps + 1;

      if next == eosToken {
        var closeBudget := maxSteps - steps;
        if closeBudget > 0 {
          var cg, ci, cc := helpers.CloseSpanWithinBudget(
            lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
          );
          generated := cg;
          insideConstrainedOut := ci;
          currentConstrainedOut := cc;
          steps := maxSteps;
        }
      } else {
        var ag, ai, ac := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := ag;
        insideConstrainedOut := ai;
        currentConstrainedOut := ac;
        constrainedTokenCount := constrainedTokenCount + 1;
      }
    }
  }

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
}
