// CSD_RATIONALE_BEGIN
// DIAGNOSIS: The best attempt (42.9%) generates free reasoning, then forces a constrained span.
// The 57.1% failures all show the same pattern:
//   1. Model generates good reasoning with organic spans (tracked via EnterObservedConstrainedSpan)
//   2. Model generates a TINY final organic span like "<<n>>" or "<<1>>"
//   3. Our forced span fires AFTER this tiny span, inheriting bad context
//   4. The model generates the same tiny trivial answer in our forced span
//
// The root cause is: when the model's final organic span is tiny (like "<<n>>"),
// our forced span appears right after it, and the model copies that trivial content.
//
// KEY INSIGHT from diagnostics:
// - "tiny_span_dominant": 37/49 examples - the final valid span is tiny
// - The model IS generating correct intermediate formulas but then appending trivial spans
// - Our forced span copies the tiny trivial answer from context
//
// CORE FIX: Do NOT use EnterObservedConstrainedSpan for organic "<<" tokens.
// Let ALL organic spans be purely free text (unconstrained).
// Only OUR forced span uses hard parser control.
// This way:
// - All organic "<<n>>" and "<<n + n*mult>>" are just free text tokens
// - Our forced span fires at ~70% budget or at EOS
// - The forced span context includes all the model's organic reasoning
// - BUT: the forced span still appears after "<<n>>" in context, so model might repeat "n"
//
// ADDITIONAL FIX: After forcing the span open, use AppendTaskGuidance-style context
// to redirect the model away from trivial answers.
//
// BETTER FIX: Force the span BEFORE the model writes its final organic span.
// Strategy: fire the forced span at ~60% of budget (earlier), so the model
// hasn't yet written its final "<<n>>" tiny span.
//
// Even better: use a TWO-TRIGGER approach:
// - Primary trigger: EOS detection during free generation (model thinks it's done)
//   → Force open a constrained span immediately
// - Secondary trigger: 60% budget reached
//
// This way the forced span fires BEFORE the model's trailing tiny span.
//
// ALSO: Inside the forced span, use RepetitionPenaltyStep to penalize commonly
// repeated short tokens, pushing toward the full formula.
//
// But RepetitionPenaltyStep penalizes ALL tokens in generated, including formula tokens.
//
// BETTER: Use AdaptiveConstrainedStepWithPenalties where we penalize short/trivial tokens.
// We can detect the "last token before >>" (likely "n" or "1") and penalize it.
//
// PLAN:
// Phase 1: Generate free text. DO NOT enter constrained mode for organic "<<" tokens.
//   Stop at EOS or at 60% budget.
// Phase 2: Force ONE constrained span.
//   - OpenConstrainedSpan
//   - First token: use SafePenalizedConstrainedStep to penalize the last token before ">>"
//     (to avoid trivially repeating the tiny organic span content)
//   - Subsequent tokens: AdaptiveConstrainedStep
//   - Close when complete
//   - CloseSpanWithinBudget for remaining budget
//
// The critical difference from the best attempt:
// - Trigger at 60% instead of 70% (fires before the model's final tiny span)
// - NO EnterObservedConstrainedSpan (organic spans stay as free text)
// - Penalize the last organic span content in the forced span
//
// This should prevent the "forced span after tiny organic span" failure pattern
// by triggering BEFORE the model writes its final tiny span.
//
// CONCERN: If we trigger at 60%, the model may not have finished reasoning yet.
// The model's reasoning is typically 30-50 free steps, then organic spans.
// With budget=900 and 60% = 540 steps, this is WAY more than enough for reasoning.
// Actually wait - stepTokenBudget=1 and maxSteps=900. So 60% = 540 steps.
// The model typically generates ~200 tokens total.
// At 60% = 540, we've well exceeded the model's natural output.
// The model will have already hit EOS before 540 steps in most cases.
//
// So the PRIMARY trigger is EOS detection (the model finishes naturally),
// and the secondary trigger is budget exhaustion.
//
// The key fix: trigger on EOS BEFORE the model writes its final organic tiny span.
// But we can't prevent the model from writing "<<n>>" unless we control all generation.
//
// ALTERNATIVE APPROACH (cleaner): Just use ManagedStep or GenerateWithManagedSpan.
// These helpers handle span lifecycle internally. Let's try GenerateWithManagedSpan
// since it's specifically designed for this pattern.
//
// GenerateWithManagedSpan: "outside a span it samples freely until EOS or <<;
// inside a span it closes when complete, otherwise advances with AdaptiveConstrainedStep"
// This is essentially what our best attempt does, but encapsulated.
//
// The problem is: GenerateWithManagedSpan will handle ALL organic spans including the
// tiny final "<<n>>" span, which is the same failure mode.
//
// FINAL APPROACH: Keep the best attempt's structure (free reasoning + forced span),
// but with these changes:
// 1. DO NOT use EnterObservedConstrainedSpan - let all organic spans be free text
// 2. Trigger forced span at EOS (primary) or 65% budget (secondary)
// 3. In the forced span, use BoostValidGroups + AdaptiveConstrainedStep
// 4. After forced span closes, BREAK (don't continue)
//
// The key improvement: by NOT entering constrained mode for organic "<<" tokens,
// the model writes ALL its reasoning (including its organic "<<n>>") as free text,
// then hits EOS. We catch that EOS and force OUR span. The context includes the
// model's organic spans as free text, which gives good context for our forced span.
//
// But the context still ends with "<<n>>." which biases the forced span toward "n".
//
// FINAL FINAL APPROACH: Use SafePenalizedConstrainedStep for the FIRST few tokens
// of the forced span, penalizing the lastTokenBefore(">>") to avoid trivial repeat.
// Then switch to AdaptiveConstrainedStep for the rest.
//
// This is what the previous attempt (attempt 45) tried and got 6.1% (worse!).
// The regression happened because SafePenalizedConstrainedStep broke something.
//
// Let me look at what changed between 42.9% (attempt 12) and 6.1% (attempt 45):
// - Attempt 45 added SafePenalizedConstrainedStep which requires the token to be
//   in lm.Tokens. But lastTok might not be in lm.Tokens! This caused issues.
//   Wait - the API says "safe" variants handle non-vocabulary tokens gracefully.
//   Actually, SafePenalizedConstrainedStep isn't in the helper list!
//   The helper list has: PenalizedConstrainedStep (NOT safe), AdaptiveConstrainedStepWithPenalties.
//
// In attempt 45, the code tried to call "helpers.SafePenalizedConstrainedStep" which
// doesn't exist in the helper list! This would cause a compilation error or runtime issue.
// Actually it says "Result: accuracy 6.1%, syntax 95.9%" - compiled but got bad results.
// Maybe the helper isn't valid but compiles... or falls through.
//
// The helper list includes:
// - PenalizedConstrainedStep (requires tokens in lm.Tokens)
// - AdaptiveConstrainedStepWithPenalties (with safe handling?)
//
// CORRECT APPROACH: Use AdaptiveConstrainedStepWithPenalties for the forced span.
// It has: "same adaptive group boosts as AdaptiveConstrainedStep, plus safe token
// penalties before the hard mask". The penaltyTokens parameter - it's "safe" in the
// sense that it filters tokens through the vocabulary internally? Not clear.
//
// Actually looking at the API: "Mechanics: GenerateLogits, conditional BoostValidGroups,
// SafePenalizeTokenLogits, MaskValidNextAndEos, ChooseNextToken." - uses SafePenalize,
// so it should handle tokens not in vocabulary.
//
// IMPLEMENTATION PLAN:
// 1. Free generation loop, NO EnterObservedConstrainedSpan
// 2. At EOS or 65% budget: force span via OpenConstrainedSpan
// 3. In forced span, first token: use AdaptiveConstrainedStepWithPenalties
//    with penaltyTokens = [lastTokBefore(">>")]
// 4. Subsequent tokens: AdaptiveConstrainedStep
// 5. Close when complete, CloseSpanWithinBudget for budget pressure
// 6. Break after forced span closes
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//   Outside-span branch (no EnterObservedConstrainedSpan):
//     We never enter constrained mode in free generation (next == "<<" is just appended
//     as free text, insideConstrainedOut stays false). The implication remains vacuously
//     true throughout free generation.
//
//   OpenConstrainedSpan (forced span entry):
//     Sets insideConstrainedOut := true, currentConstrainedOut := [].
//     parser.IsValidPrefix([]) holds by precondition. Invariant established.
//
//   CloseConstrainedSpan:
//     Sets insideConstrainedOut := false, making the implication vacuously true.
//     currentConstrainedOut := []. Invariant holds.
//
//   CloseSpanWithinBudget:
//     Postcondition: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut).
//     Invariant preserved by helper contract.
//
//   AdaptiveConstrainedStepWithPenalties and AdaptiveConstrainedStep:
//     Both apply hard mask to parser-valid next tokens + EOS.
//     If next != eosToken, then parser.IsValidPrefix(currentConstrainedOut + [next]) holds.
//     AppendConstrainedToken extends currentConstrainedOut by [next], preserving validity.
//
// progress: |generated| <= |generatedPrefix| + steps
//
//   UnconstrainedStep (free text, no span entry):
//     steps += 1, generated += [next] (at most 1 token for non-EOS).
//     EOS: steps += 1, no append, forced span branch taken.
//     Invariant: |generated| <= |generatedPrefix| + steps preserved.
//
//   OpenConstrainedSpan:
//     steps += 1, generated grows by exactly 1 ("<<").
//     Invariant preserved.
//
//   CloseConstrainedSpan:
//     steps += 1, generated grows by at most 1 (">>").
//     Invariant preserved.
//
//   CloseSpanWithinBudget with closeBudget = maxSteps - steps:
//     Postcondition: |generatedOut| <= |generated| + closeBudget.
//     After steps := maxSteps:
//     |generated| = |old_generated| + delta <= |generatedPrefix| + old_steps + closeBudget
//                 = |generatedPrefix| + maxSteps = |generatedPrefix| + steps.
//     Invariant preserved.
//
//   AdaptiveConstrainedStep/WithPenalties + AppendConstrainedToken:
//     steps += 1, AppendConstrainedToken appends exactly 1 token to generated.
//     |generated| <= |generatedPrefix| + steps. Invariant preserved.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve this math problem step by step. Show your work with intermediate expressions in << >> delimiters. The final << >> span must contain the complete final answer as a mathematical expression (e.g. <<n * (mult + 1)>> not just <<n>>).";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;
var narrowThreshold: nat := 12;
// Trigger forced span at 65% of budget or at EOS
var freeStepsTarget: nat := (maxSteps * 65) / 100;
var forcedFinalSpan: bool := false;

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
    // Check if we should force the final constrained span
    var shouldForce := !forcedFinalSpan && (steps >= freeStepsTarget || maxSteps - steps <= 5);
    if shouldForce && maxSteps - steps >= 3 {
      // Force open a constrained span for the final answer
      var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
      generated := og;
      insideConstrainedOut := oi;
      currentConstrainedOut := oc;
      steps := steps + 1;
      forcedFinalSpan := true;
    } else {
      // Free generation for reasoning - NO EnterObservedConstrainedSpan
      var next := helpers.UnconstrainedStep(lm, prompt, generated);
      steps := steps + 1;
      if next == eosToken {
        // Model wants to EOS - force a final constrained span if not yet done
        if !forcedFinalSpan && maxSteps - steps >= 3 {
          var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
          generated := og;
          insideConstrainedOut := oi;
          currentConstrainedOut := oc;
          steps := steps + 1;
          forcedFinalSpan := true;
          // Continue loop to fill the span
        } else {
          break;
        }
      } else {
        // Append token as free text (including any organic "<<" tokens)
        generated := generated + [next];
        // NOTE: We deliberately do NOT call EnterObservedConstrainedSpan here.
        // Organic "<<" tokens become part of free text, avoiding the tiny-span failure.
      }
    }
  } else {
    // Inside the forced constrained span
    if parser.IsCompletePrefix(currentConstrainedOut) {
      // Span is complete: close it
      var cg, ci, cc := helpers.CloseConstrainedSpan(
        lm, parser, generated, currentConstrainedOut
      );
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := steps + 1;
      // After closing the forced span, we're done
      break;
    } else if maxSteps - steps <= 4 {
      // Near budget end: use CloseSpanWithinBudget to finish gracefully
      var closeBudget := maxSteps - steps;
      var cg, ci, cc := helpers.CloseSpanWithinBudget(
        lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
      );
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := maxSteps;
    } else {
      // Generate next token inside the forced span
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next;
      if |currentConstrainedOut| == 0 {
        // First token in the forced span: penalize the last organic span's content
        // to avoid trivially repeating a short formula
        var lastTok, foundLast := helpers.LastTokenBefore(generated, ">>");
        if foundLast {
          // Use AdaptiveConstrainedStepWithPenalties to discourage trivial repetition
          next := helpers.AdaptiveConstrainedStepWithPenalties(
            lm, parser, constrainedPrompt, currentConstrainedOut,
            validTokenGroups, 4.0, [lastTok], 5.0, narrowThreshold, eosToken
          );
        } else {
          next := helpers.AdaptiveConstrainedStep(
            lm, parser, constrainedPrompt, currentConstrainedOut,
            validTokenGroups, 4.0, narrowThreshold, eosToken
          );
        }
      } else {
        next := helpers.AdaptiveConstrainedStep(
          lm, parser, constrainedPrompt, currentConstrainedOut,
          validTokenGroups, 4.0, narrowThreshold, eosToken
        );
      }
      steps := steps + 1;
      if next == eosToken {
        // EOS inside span: close the span within remaining budget
        var remaining := maxSteps - steps;
        if remaining >= 1 {
          var cg2, ci2, cc2 := helpers.CloseSpanWithinBudget(
            lm, parser, prompt, generated, currentConstrainedOut, eosToken, remaining
          );
          generated := cg2;
          insideConstrainedOut := ci2;
          currentConstrainedOut := cc2;
          steps := steps + remaining;
        }
        break;
      } else {
        var ag, ai, ac := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := ag;
        insideConstrainedOut := ai;
        currentConstrainedOut := ac;
      }
    }
  }
}

cost := steps;

