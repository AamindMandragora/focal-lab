// CSD_RATIONALE_BEGIN
// DIAGNOSIS: 5 unterminated spans remain. The unterminated cases show the model
// generates partial arithmetic expressions like "1 - (1/{" or "t * {" or
// "n1 * price" that contain template variables like "{" which may confuse the
// parser. The parser sees "{" and can't extend to a valid/complete expression.
//
// The 5 unterminated spans are the difference between 89.8% and 92% syntax.
// We need to close 2 more spans (2/49 = 4.1pp improvement needed = 2.2pp gap).
//
// KEY INSIGHT from rollouts:
// The model generates expressions like "1 - (1/{..." where "{" is a template
// variable placeholder. The parser can't close this. When we hit a dead end
// (DeadEndDetection fires) or the parser is stuck on "{", the rollback to
// complete-or-empty should be triggered.
//
// Current attempt 6 adds RollbackConstrainedToComplete as a final fallback,
// which improved syntax from 87.8% to 89.8% (2 more spans closed).
// We need 1-2 more spans to close.
//
// ANALYSIS of remaining failures:
// Looking at the rollouts, the unterminated spans happen when:
// 1. The model has generated partial content inside << >>
// 2. The content reaches a template var like "{"
// 3. DeadEndDetection eventually fires (but maybe too late - the token "{" was
//    already appended before dead-end is detected)
// 4. Our 6x rollback + RollbackConstrainedToComplete fires
// 5. But the rollback to complete is empty (no valid complete prefix from the partial)
// 6. Then we try ONE ConstrainedStep to regenerate
// 7. That step generates another bad token, we end up stuck again
// 8. Budget exhausted, span stays open
//
// The issue: after RollbackConstrainedToComplete returns empty, we need to
// REGENERATE more aggressively. Currently we only try ONE ConstrainedStep.
// After that one step, if not complete, we loop back to the top of the while.
// But the span is still OPEN (insideConstrainedOut=true) and currentConstrainedOut=[].
// The loop should then try to generate a complete expression from scratch.
//
// WAIT - if after RollbackConstrainedToComplete we get empty, and then we loop
// back to the top, we're in the `else` branch (not complete, not dead end with 0 tokens).
// Then ConfidenceGatedStep is called and appends tokens one by one. This IS regenerating
// from scratch. So the loop DOES regenerate after rollback to empty.
//
// The problem must be that the loop runs out of steps (steps >= maxSteps) while
// the span is still open. The post-loop cleanup does a RollbackConstrainedToComplete,
// but if the span is only partially built (not a complete valid expression),
// it returns empty, and we can't close it.
//
// REAL FIX: When a span has been open for a long time and the model keeps generating
// partial expressions, we need a more aggressive close mechanism.
//
// APPROACH: Use RollbackAndContinue which does rollback + regeneration with
// dead-end avoidance. This is specifically designed for this case.
// closeReserve=2 (need 2 steps: one for close, one buffer).
// maxRetries=5 to avoid dead ends.
//
// But wait - RollbackAndContinue requires maxSteps steps from the HELPER level,
// and maxSteps must be <= remaining steps. Let me think about this carefully.
//
// Actually, the key issue may be SIMPLER: the 5 unterminated spans are in the
// "slow" time band (mode_B: gen_time_band slow 80%). These are running close
// to the 120s wall-clock budget. The slowness is due to the complex expressions
// the model tries to generate.
//
// Strategy: detect when we're running out of steps and force-close aggressively.
// If remaining steps <= spanMaxTokens + 2, force a constrained close attempt
// immediately via RollbackConstrainedToComplete + close.
//
// The "remaining <= 65" check in the unconstrained phase is for opening spans.
// We need a similar check INSIDE the span: if remaining <= 3, rollback to complete
// and close immediately, don't try to generate more tokens.
//
// REVISED PLAN:
// 1. Keep attempt 6's structure (it's already the best)
// 2. Add a check at the START of each inside-span iteration:
//    if steps + 3 >= maxSteps: force rollback to complete + close
// 3. This handles the case where we're running out of budget inside a span
// 4. Use a slightly longer span token budget check (remaining <= 4 instead of 3)
//
// Also: The post-loop cleanup needs to be more aggressive.
// After the while loop exits (steps >= maxSteps), try hard to close.
// The cleanup should use remaining budget (steps might be < maxSteps due to breaks).
//
// ADDITIONAL: In the failing examples, the model generates things like
// "<<1 - (1/{..." with template variables that are parser-valid individual tokens
// but lead to parser dead ends. The SPAN is opened, then the model puts in "{".
// DeadEndDetection should catch "{" causing a dead end, but maybe the parser
// allows "{" as a valid token in some arithmetic grammar?
//
// The spans that work are like "<<n * m>>" or "<<n1 + mult * 1 + 1>>".
// The spans that fail are like "<<1 - (1/{..." - these have malformed template vars.
//
// CRUCIAL INSIGHT: the model is generating INSIDE the constrained span with
// template variables like {target}, {frac1}, etc. The arithmetic parser may
// accept "1 - (1 / " but then sees "{" which is likely valid as a variable name
// or invalid, causing the constraint to get stuck.
//
// The real solution: force the span to close quickly (within 8-10 tokens) using
// a hard budget, and at token 8-10, aggressively roll back to complete.
// This is spanMaxTokens. We have it at 12. Maybe reducing to 8 would help
// the failing cases close faster.
//
// BUT: attempt 5 with spanMaxTokens=10 got worse (syntax dropped to 75.5%!).
// And attempt 1 with spanMaxTokens=12 got 42.9%/87.8%.
// So reducing spanMaxTokens hurts.
//
// Wait, attempt 5's regression was because of the 10x rollback causing spinning.
// With the CURRENT fixed rollback (RollbackConstrainedToComplete as safety net),
// reducing spanMaxTokens might be safe.
//
// FINAL STRATEGY: Keep attempt 6 but:
// 1. Add an early-close guard at the top of the inside-span block:
//    if maxSteps - steps <= 2: immediately RollbackConstrainedToComplete + close
//    This prevents running out of budget with open spans
// 2. Reduce spanMaxTokens to 10 (to trigger rollback earlier before expressions
//    get too long and complex like "1 - (1/{...")
//    The previous attempt 5 regressed because of 10x suffix rollback + ToComplete.
//    With only 6x suffix + ToComplete as final (attempt 6's approach), reducing
//    to 10 should be safer.
// 3. Keep everything else the same as attempt 6
//
// ACTUALLY - looking more carefully at mode_B: 5 samples, gen_time_band slow (80%).
// These are slow because the parser is doing 49329 rollbacks per 49 examples = 1006 per example.
// With slow=80% for mode_B, these 4 examples are near the time limit.
// They're failing because they use more steps, not fewer.
//
// Let me reconsider. The 5 unterminated spans are using LOW token budget (100%).
// Wait - mode_B has token_budget_band: low (100%). "Low" budget means they didn't
// use many tokens. So they terminated EARLY, not late!
//
// They terminated early because the span never closed and the budget ran out?
// Or they're "slow" in runtime but "low" in tokens? Low tokens with slow runtime
// means each token took a long time (complex model calls).
//
// If token_budget_band is low, they used fewer tokens than average (avg=113).
// If gen_time_band is slow, they were slow per token. 
// These might be the examples where the model gets stuck in an expensive 
// rollback loop (49329 RollbackToValidPrefix calls = very expensive for parser queries).
//
// SYNTHESIS: The 5 unterminated spans have LOW tokens (ran out of generation before closing)
// AND slow runtime (lots of parser work). They're hitting a parser dead-end loop
// where DeadEndDetection fires repeatedly, we roll back, generate again, dead end,
// roll back, etc. Eventually budget exhausted.
//
// The fix is to detect when we've been stuck in a dead-end loop and immediately
// escape via RollbackConstrainedToComplete + close (or abandon the span and continue unconstrained).
//
// With the current strategy's spanTokensUsed tracking, after spanMaxTokens=12 tokens
// we trigger the rollback. But if the rollback+regen cycle keeps hitting dead ends
// WITHOUT advancing spanTokensUsed (because DeadEndDetection fires before appending),
// we'll loop indefinitely.
//
// KEY BUG: In the dead-end/over-budget branch, after RollbackConstrainedToComplete:
// - If empty, we do ONE ConstrainedStep
// - If that's not EOS and valid, we append and set spanTokensUsed=1
// - NEXT ITERATION: span has 1 token, not complete, not dead end → ConfidenceGatedStep
// - ConfidenceGatedStep generates another token → append → spanTokensUsed=2
// - ...eventually spanTokensUsed hits 12 again → rollback → empty → one step → etc.
// - This cycle uses steps each time, so it IS making progress toward running out of budget
// - After enough cycles, steps >= maxSteps and loop exits with open span
//
// The REAL fix is: after RollbackConstrainedToComplete returns empty,
// if we've been in the rollback branch, this means the model can't generate
// a valid complete expression from this context. We should CLOSE the span with
// whatever we have (or abandon it). 
//
// Actually we CAN'T close an empty span (not complete). We need at least one token
// for the expression to be complete (like a number "42").
//
// The problem is the arithmetic grammar may require a valid number/expression.
// If the model keeps generating template vars like "{target}" the parser sees
// "{" and may have no valid completion.
//
// ULTIMATE FIX: Track how many times we've done RollbackConstrainedToComplete.
// After 2-3 times hitting empty, GIVE UP on constrained and just do an unconstrained
// close. But we can't do an unconstrained close (CloseConstrainedSpan requires complete).
//
// OR: Use UnconstrainedStep to get the close token ">>" directly.
// After 2 failed rollback attempts, exit the constrained span by other means.
// We can set insideConstrainedOut=false directly? No, we can't do that in Dafny
// without a helper that handles the state.
//
// ALTERNATIVE: Use a "rollback budget" counter. After N failed rollback cycles,
// set a flag that forces smaller span generation with very short limit (2-3 tokens).
// The short limit forces closure of simple expressions like single numbers.
//
// PRACTICAL APPROACH: Add a `rollbackCount` variable. After each RollbackConstrainedToComplete
// that results in empty, increment rollbackCount. If rollbackCount >= 2, reduce
// spanMaxTokens to 3. This forces very short expressions (like single numbers)
// which are more likely to be complete.
//
// With spanMaxTokens=3, the model must produce a complete expression in ≤3 tokens.
// For GSM8k answers like "42", "150", "2.5", this is 1-2 tokens. So it would work.
//
// IMPLEMENTING THIS: Simple and targeted.
// - rollbackCount tracks failed rollback-to-complete cycles
// - After rollbackCount >= 2: spanMaxTokens_local = 3 (aggressive close)
// - This prevents the infinite loop without reducing accuracy for normal cases
//
// Let me estimate the effect: For the 5 failing examples, we'd force close
// within 3 tokens after 2 failed attempts. The answer might be wrong (wrong number)
// but at least the span closes. So syntax improves by 5 examples = +10.2pp.
// Accuracy might drop slightly if some correct answers were in those 5 examples.
// But mode_B examples are all wrong anyway (syntax_invalid_wrong), so no accuracy loss.
//
// EXPECTED RESULT: 89.8% + 10.2% = ~100% syntax? No, that's too optimistic.
// The 5 mode_B examples represent unterminated spans. Closing them = syntax valid.
// 44 valid + 5 newly valid = 49/49 = 100% syntax. But wait, "balanced" spans show
// 39/49, "unterminated" shows 10/49. The 10 unterminated - 5 that still close =
// 5 remaining? The metric "Syntax Rate: 89.8%" = 44/49 valid.
// Closing all 5 remaining unterminated = 49/49 = 100% syntax!
// That's MORE than needed (92%). 
//
// REVISED TARGET: Close those 5 unterminated spans. Strategy:
// 1. Track rollbackCount (how many times we've done full rollback-to-empty)
// 2. After rollbackCount >= 2 AND !complete, use spanMaxTokens=4 (aggressive)
// 3. This forces the model to produce short expressions that complete quickly
// 4. Better post-loop cleanup with multiple constrained steps
//
// IMPLEMENTATION: Keep attempt 6's structure exactly, add rollbackCount tracking.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//   Outside span (EnterObservedConstrainedSpan): sets currentConstrainedOut:=[],
//     parser.IsValidPrefix([]) holds by precondition.
//   Outside span (OpenConstrainedSpan): sets currentConstrainedOut:=[],
//     parser.IsValidPrefix([]) holds by precondition.
//   Inside span (IsCompletePrefix): CloseConstrainedSpan sets insideConstrainedOut:=false,
//     making the implication vacuously true.
//   Inside span (early-close guard): RollbackConstrainedToComplete postcondition gives
//     IsCompletePrefix or empty; if complete, CloseConstrainedSpan makes implication vacuous;
//     if empty, IsValidPrefix([]) by precondition.
//   Inside span (6x RollbackConstrainedSuffix): each call postcondition gives IsValidPrefix(rolled).
//   Inside span (RollbackConstrainedToComplete): postcondition gives IsValidPrefix or empty.
//   Inside span (ConfidenceGatedStep + AppendConstrainedToken): ConfidenceGatedStep uses
//     hard mask when LM's top token is not parser-valid; AppendConstrainedToken postcondition
//     guarantees IsValidPrefix preserved.
//   Inside span (ConstrainedStep + AppendConstrainedToken): hard mask guarantees IsTokenValidNext;
//     AppendConstrainedToken preserves IsValidPrefix.
//
// progress: |generated| <= |generatedPrefix| + steps
//
//   UnconstrainedChunk: steps+=stepsUsed, |generated| grows by at most stepsUsed tokens. ✓
//   OpenConstrainedSpan: steps+=1, |generated| grows by exactly 1 ("<<"). ✓
//   EnterObservedConstrainedSpan: steps+=0, |generated| unchanged. ✓
//   CloseConstrainedSpan: steps+=1, |generated| grows by at most 1 (">>"). ✓
//   ConfidenceGatedStep+AppendConstrainedToken: steps+=1, |generated| grows by 1. ✓
//   ConstrainedStep+AppendConstrainedToken: steps+=1, |generated| grows by 1. ✓
//   RollbackConstrainedSuffix/ToComplete: cost+0, |generated| may shrink (bound preserved). ✓
//   All step-consuming calls guarded by steps < maxSteps check. ✓
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve step by step. Write the final numeric answer inside << >>. Example: <<42>>. Keep the answer expression short.");

var steps: nat := 0;
var freeChunkSize: nat := 25;
var spanTokensUsed: nat := 0;
var spanMaxTokens: nat := 12;
var hasSeenOpenSpan: bool := insideConstrained;
var rollbackCount: nat := 0;

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
    var remaining: nat := maxSteps - steps;
    if remaining <= 65 && !hasSeenOpenSpan && remaining > 2 {
      var g2, i2, c2 := helpers.OpenConstrainedSpan(lm, generated);
      generated := g2;
      insideConstrainedOut := i2;
      currentConstrainedOut := c2;
      steps := steps + 1;
      spanTokensUsed := 0;
      rollbackCount := 0;
      hasSeenOpenSpan := true;
    } else {
      var chunkBudget: nat := if remaining < freeChunkSize then remaining else freeChunkSize;
      if chunkBudget == 0 {
        break;
      }
      var chunkGenerated, stoppedOnOpenSpan, stoppedOnEos, stepsUsed :=
        helpers.UnconstrainedChunk(lm, prompt, generated, chunkBudget, "<<", eosToken);
      generated := chunkGenerated;
      steps := steps + stepsUsed;
      if stoppedOnEos {
        if !hasSeenOpenSpan && steps + 3 <= maxSteps {
          var g2, i2, c2 := helpers.OpenConstrainedSpan(lm, generated);
          generated := g2;
          insideConstrainedOut := i2;
          currentConstrainedOut := c2;
          steps := steps + 1;
          spanTokensUsed := 0;
          rollbackCount := 0;
          hasSeenOpenSpan := true;
        } else {
          break;
        }
      } else if stoppedOnOpenSpan {
        var g2, i2, c2 := helpers.EnterObservedConstrainedSpan(lm, generated);
        generated := g2;
        insideConstrainedOut := i2;
        currentConstrainedOut := c2;
        spanTokensUsed := 0;
        rollbackCount := 0;
        hasSeenOpenSpan := true;
      }
    }
  } else {
    // Inside constrained span
    var remaining: nat := maxSteps - steps;

    // Early-close guard: if very little budget left, force close immediately
    if remaining <= 3 {
      var gFinal, cFinal := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
      generated := gFinal;
      currentConstrainedOut := cFinal;
      if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
        var g2, i2, c2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
        generated := g2;
        insideConstrainedOut := i2;
        currentConstrainedOut := c2;
        steps := steps + 1;
      }
      break;
    } else if parser.IsCompletePrefix(currentConstrainedOut) {
      var g2, i2, c2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := g2;
      insideConstrainedOut := i2;
      currentConstrainedOut := c2;
      steps := steps + 1;
      spanTokensUsed := 0;
      rollbackCount := 0;
    } else {
      // Determine effective max tokens based on rollback count (adaptive)
      var effectiveMax: nat := if rollbackCount >= 2 then 4 else spanMaxTokens;

      var isDeadEnd := helpers.DeadEndDetection(parser, currentConstrainedOut, 1);
      if isDeadEnd || spanTokensUsed >= effectiveMax {
        // Rollback up to 6 times to find a closeable state
        var gR1, cR1 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
        generated := gR1;
        currentConstrainedOut := cR1;
        if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
          var gR2, cR2 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
          generated := gR2;
          currentConstrainedOut := cR2;
        }
        if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
          var gR3, cR3 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
          generated := gR3;
          currentConstrainedOut := cR3;
        }
        if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
          var gR4, cR4 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
          generated := gR4;
          currentConstrainedOut := cR4;
        }
        if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
          var gR5, cR5 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
          generated := gR5;
          currentConstrainedOut := cR5;
        }
        if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
          var gR6, cR6 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
          generated := gR6;
          currentConstrainedOut := cR6;
        }
        spanTokensUsed := 0;
        if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
          var g2, i2, c2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
          generated := g2;
          insideConstrainedOut := i2;
          currentConstrainedOut := c2;
          steps := steps + 1;
          rollbackCount := 0;
        } else if steps < maxSteps {
          // Not complete after 6x rollback: use RollbackConstrainedToComplete for safety
          var gFinal, cFinal := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
          generated := gFinal;
          currentConstrainedOut := cFinal;
          if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
            var g2, i2, c2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
            generated := g2;
            insideConstrainedOut := i2;
            currentConstrainedOut := c2;
            steps := steps + 1;
            rollbackCount := 0;
          } else {
            // Empty after full rollback: increment rollbackCount and try ConstrainedStep
            rollbackCount := rollbackCount + 1;
            var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
            var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
            steps := steps + 1;
            if next == eosToken {
              // EOS after full rollback: give up on this span context
              // Try one more RollbackConstrainedToComplete and close if possible
              var gFinal2, cFinal2 := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
              generated := gFinal2;
              currentConstrainedOut := cFinal2;
              if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
                var g3, i3, c3 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
                generated := g3;
                insideConstrainedOut := i3;
                currentConstrainedOut := c3;
                steps := steps + 1;
                rollbackCount := 0;
              }
            } else {
              var g2, i2, c2 := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
              generated := g2;
              insideConstrainedOut := i2;
              currentConstrainedOut := c2;
              spanTokensUsed := spanTokensUsed + 1;
              if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
                var g3, i3, c3 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
                generated := g3;
                insideConstrainedOut := i3;
                currentConstrainedOut := c3;
                steps := steps + 1;
                rollbackCount := 0;
                spanTokensUsed := 0;
              }
            }
          }
        }
      } else {
        // Normal constrained step using ConfidenceGatedStep for accuracy
        var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
        var next, wasConstrained := helpers.ConfidenceGatedStep(
          lm, parser, constrainedPrompt, currentConstrainedOut, eosToken
        );
        steps := steps + 1;
        if next == eosToken {
          // Rollback to find complete prefix
          var gR1, cR1 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
          generated := gR1;
          currentConstrainedOut := cR1;
          if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
            var gR2, cR2 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
            generated := gR2;
            currentConstrainedOut := cR2;
          }
          if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
            var gR3, cR3 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
            generated := gR3;
            currentConstrainedOut := cR3;
          }
          if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
            var gR4, cR4 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
            generated := gR4;
            currentConstrainedOut := cR4;
          }
          if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
            var gR5, cR5 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
            generated := gR5;
            currentConstrainedOut := cR5;
          }
          // Final fallback: RollbackConstrainedToComplete guarantees complete or empty
          if !parser.IsCompletePrefix(currentConstrainedOut) {
            var gFinal, cFinal := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
            generated := gFinal;
            currentConstrainedOut := cFinal;
            if !parser.IsCompletePrefix(currentConstrainedOut) {
              rollbackCount := rollbackCount + 1;
            }
          }
          spanTokensUsed := 0;
          if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
            var g2, i2, c2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
            generated := g2;
            insideConstrainedOut := i2;
            currentConstrainedOut := c2;
            steps := steps + 1;
            rollbackCount := 0;
          }
          // If not complete (empty), continue loop to regenerate
        } else {
          var g2, i2, c2 := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
          generated := g2;
          insideConstrainedOut := i2;
          currentConstrainedOut := c2;
          spanTokensUsed := spanTokensUsed + 1;
        }
      }
    }
  }
}

// Post-loop cleanup: if we exited with an open span, try to close it
if insideConstrainedOut && steps < maxSteps {
  // First try RollbackConstrainedToComplete as the most reliable mechanism
  var gFinal, cFinal := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
  generated := gFinal;
  currentConstrainedOut := cFinal;
  if !parser.IsCompletePrefix(currentConstrainedOut) && steps + 1 < maxSteps {
    // Empty after rollback - try one constrained step
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
    steps := steps + 1;
    if next != eosToken {
      var g2, i2, c2 := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
      generated := g2;
      insideConstrainedOut := i2;
      currentConstrainedOut := c2;
      // Check if one token was enough for complete
      if !parser.IsCompletePrefix(currentConstrainedOut) {
        var gF2, cF2 := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
        generated := gF2;
        currentConstrainedOut := cF2;
      }
    }
  }
  if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
    var g2, i2, c2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := g2;
    insideConstrainedOut := i2;
    currentConstrainedOut := c2;
    steps := steps + 1;
  }
}

cost := steps;

