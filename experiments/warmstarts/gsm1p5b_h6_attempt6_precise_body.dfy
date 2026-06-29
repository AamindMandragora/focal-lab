// CSD_RATIONALE_BEGIN
// DIAGNOSIS: Best attempt (1) achieved 42.9% accuracy, 87.8% syntax.
// Attempt 5 regressed to 26.5% / 75.5% by being too aggressive with rollback
// (spanMaxTokens=10, 10 rollbacks). Attempt 4 was 36.7% / 83.7% with spanMaxTokens=15.
//
// The core problem: 12/49 unterminated spans in attempt 5, 6-8 in attempts 1/4.
// Attempt 1 used ConfidenceGatedStep (less reliable parser control) but got
// 42.9% accuracy. The accuracy regression in attempts 3-5 was partly due to
// over-aggressive rollback cutting good expressions short.
//
// KEY OBSERVATIONS from feedback:
// 1. "Avg complete visible spans/example: 2.08" - multiple spans per example!
//    The model opens MULTIPLE << spans. The last one is the answer.
//    With 2.08 spans, the model is doing intermediate calculations too.
// 2. "Examples with balanced visible spans: 15/49" but "37/49 have complete spans"
//    This means many examples have more opens than closes (unterminated).
// 3. "RollbackToValidPrefix=37193" = 37193/49 = 759 rollbacks per example!
//    That's insane - something is spinning in rollback loops.
//
// Wait, 37193 rollback calls for 49 examples = 759 per example on average.
// Each rollback removes one token, so with ~4 token spans, we're triggering
// rollback ~190 times per span. This is the SPINNING problem.
//
// The spinning happens because:
// 1. Model generates token T inside span
// 2. spanTokensUsed >= spanMaxTokens=10 OR isDeadEnd fires
// 3. We do 10x RollbackConstrainedSuffix 
// 4. Back to empty span, ConstrainedStep picks token T again (same context!)
// 5. Repeat -> infinite loop consuming the budget
//
// With 49 examples * 87 avg tokens = 4263 total tokens, but 37193 rollbacks...
// The rollbacks are NOT consuming tokens, so this isn't the budget issue.
// But 759 rollback calls per example * 0.21s/token * 49 examples = slowness.
//
// Actually, "Runtime per generated token: 0.21s" and "87.35 tokens/example avg"
// = 18.3s per example avg. With 120s budget per example, we're using 15% of the
// budget just on token generation. The rollbacks are free (no LM calls), so they
// don't slow us down much.
//
// The spinning is the REAL problem: the model opens a new << after each >>
// (because UnconstrainedChunk generates "<<" again), or the constrained content
// loops. With 2.08 spans per example, each span might be closing but then
// immediately opening again.
//
// INSIGHT: The output shows "<<n1 + n2 + n3>> balls. The total cost is <<n"
// This means:
// 1. First span "<<n1 + n2 + n3>>" closed successfully
// 2. Then the model (in unconstrained phase) wrote " balls. The total cost is "
// 3. Then model wrote "<<n" as the start of a new span
// 4. "n" was constrained, generated token "n", then hit dead end or budget
//
// So the LAST span is the problem. The model opens a final span but runs
// out of budget or hits a dead end.
//
// STRATEGY REVISION:
// The key insight: once we have ONE valid complete span (hasSeenOpenSpan && span closed),
// that's the final answer for many problems. But the model keeps opening more spans.
// For gsm_symbolic, the LAST << >> is the final answer.
//
// The fundamental tension: more spans = more attempt at the right answer,
// but more risk of the last one being unterminated.
//
// APPROACH: Use ConstrainedStep (hard mask) instead of ConfidenceGatedStep
// for all constrained generation. This ensures strong parser control.
// ConfidenceGatedStep was causing issues because when the model's top token
// is valid but leads to a dead end later, we don't mask it.
//
// Use RollbackConstrainedToComplete as the ROLLBACK mechanism (not suffix rollback)
// since it guarantees complete-or-empty. Then immediately close.
//
// Keep spanMaxTokens=12 (same as attempt 1) to allow reasonable expressions.
// After maxTokens or deadend, use RollbackConstrainedToComplete + close.
//
// CRITICAL: The 759 rollbacks/example is because RollbackConstrainedSuffix
// is called from INSIDE the loop without consuming steps. If span is empty
// and not complete, we do 10 rollbacks that all return empty, then ConstrainedStep
// picks a token, we append, then immediately rollback again = spinning.
//
// FIX: Use a single RollbackConstrainedToComplete call (guaranteed to work in O(n))
// instead of 10x RollbackConstrainedSuffix. This is attempt 3's approach but we
// also need ConfidenceGatedStep for accuracy (not in attempt 3).
//
// Wait, attempt 3 used RollbackConstrainedToComplete and got 26.5%/77.6%.
// Attempt 1 (best) used RollbackConstrainedSuffix x6 and got 42.9%/87.8%.
//
// The difference between attempts 1 and 3:
// Attempt 1: spanMaxTokens=12, 6x RollbackConstrainedSuffix, ConfidenceGatedStep
// Attempt 3: spanMaxTokens=10, RollbackConstrainedToComplete, no ConfidenceGatedStep
//
// Attempt 3 regressed BOTH accuracy AND syntax vs attempt 1. So:
// - RollbackConstrainedToComplete alone is worse than 6x RollbackConstrainedSuffix
// - Removing ConfidenceGatedStep hurts accuracy
//
// HYPOTHESIS: The 6x RollbackConstrainedSuffix in attempt 1 accidentally preserves
// more of the expression (rolls back to valid state, not necessarily complete),
// and then CONTINUES generating from that valid state. This gives the LM a chance
// to extend the valid prefix to a complete expression.
//
// RollbackConstrainedToComplete jumps too far back (to complete or empty), losing
// the partial expression that the LM was building. The LM then restarts from
// scratch and might generate the same wrong answer.
//
// So attempt 1's strategy is correct: 6x rollback to find complete/closeable state,
// then continue if not complete. The SPINNING issue is separate.
//
// Why did attempt 5 regress from attempt 1?
// Attempt 5 used 10x RollbackConstrainedSuffix (more than 6x), PLUS
// RollbackConstrainedToComplete as safety net. This caused MORE rollbacks
// (as seen: 37193 vs attempt 1's presumably much lower count).
//
// The extra RollbackConstrainedToComplete removes all partial progress,
// causing the model to restart and regenerate, leading to more spans
// and more unterminated spans.
//
// VERDICT: GO BACK TO ATTEMPT 1 STRATEGY EXACTLY.
// The best strategy is attempt 1 (42.9%/87.8%). The improvements I've tried
// have all made things worse. The remaining gap to 31%/92% is:
// - accuracy already at 42.9% > 31% ✓  
// - syntax at 87.8% < 92% ✗ (need +4.2pp)
//
// To fix 4.2pp syntax (2 more examples out of 49):
// Need to close 2 more unterminated spans without hurting accuracy.
// Attempt 1 had ~6 unterminated spans. We need 4 of them to close.
//
// MINIMAL TARGETED FIX:
// In attempt 1, the 6 unterminated spans are cases where:
// - After 6x rollback, span is not complete
// - ConstrainedStep picks EOS or dead-end token
// - Span remains open
//
// The fix: after the 6x rollback + ConstrainedStep, if span is still not complete
// AND token was EOS, use RollbackConstrainedToComplete as a LAST resort.
// This is targeted to only the "last resort" case, not the normal case.
// Only use it when:
// 1. We've already done 6x rollback (span is valid but not complete)
// 2. ConstrainedStep returned EOS (model wants to quit)
// 3. We're in the rollback/dead-end branch (not normal generation)
//
// Also: After span closes, limit further unconstrained generation to avoid
// opening new spans that might not close. Once we've seen a complete span,
// limit the unconstrained phase to just a few more tokens.
//
// Actually the issue is MULTIPLE spans. The model opens <<n1 + n2 + n3>>
// then opens another span. If we STOP after the first complete span closes
// (or after N complete spans), we'd avoid the unterminated last span.
//
// But then we might miss the correct FINAL answer which is in the last span.
// For gsm_symbolic, the evaluator uses the LAST complete span as the answer.
// So if we have multiple correct intermediate spans and one unterminated final,
// we'd get a wrong answer (from last complete = intermediate) OR no answer (no complete span).
//
// BETTER APPROACH: Limit the number of spans. After seeing 3 complete spans,
// stop unconstrained generation (or reduce freeChunkSize to 0 to stop).
// But this requires tracking span count.
//
// Actually the simplest fix: after each span closes, use a SMALLER freeChunkSize
// (e.g., 15 instead of 25) to limit how much text the model generates before
// the next span. This reduces the chance of the model opening a new span
// right before budget runs out.
//
// No - the real fix is to ensure spans that DO open also close.
// The strategy should be:
// 1. Keep attempt 1's structure
// 2. Add RollbackConstrainedToComplete as a FINAL fallback after 6x suffix rollback
//    and failed ConstrainedStep (EOS case)
// 3. Keep spanMaxTokens=12 (don't reduce - that hurts accuracy)
//
// This should fix 2-3 more spans without regressing accuracy.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//   Outside span (UnconstrainedChunk stoppedOnOpenSpan): EnterObservedConstrainedSpan
//     sets currentConstrainedOut:=[], IsValidPrefix([]) holds by precondition.
//   Outside span (OpenConstrainedSpan forced): sets currentConstrainedOut:=[],
//     IsValidPrefix([]) holds by precondition.
//   Inside span (IsCompletePrefix): CloseConstrainedSpan sets insideConstrainedOut:=false,
//     implication vacuously true.
//   Inside span (ConfidenceGatedStep + AppendConstrainedToken): ConfidenceGatedStep
//     uses hard mask when needed, ensuring IsTokenValidNext holds for the appended
//     token; AppendConstrainedToken postcondition guarantees IsValidPrefix preserved.
//   Inside span (6x RollbackConstrainedSuffix): each call's postcondition guarantees
//     IsValidPrefix(cRolled). If RollbackConstrainedToComplete is called, its
//     postcondition gives IsCompletePrefix (implies IsValidPrefix) or empty
//     (IsValidPrefix([]) by precondition).
//   ConstrainedStep + AppendConstrainedToken: hard mask ensures token valid;
//     AppendConstrainedToken preserves IsValidPrefix.
//
// progress: |generated| <= |generatedPrefix| + steps
//
//   UnconstrainedChunk: steps+=stepsUsed, |generated| grows by at most stepsUsed. ✓
//   OpenConstrainedSpan: steps+=1, |generated| grows by 1 ("<<"). ✓
//   EnterObservedConstrainedSpan: steps+=0, |generated| unchanged. ✓
//   CloseConstrainedSpan: steps+=1, |generated| grows by at most 1 (">>"). ✓
//   ConfidenceGatedStep+AppendConstrainedToken: steps+=1, |generated| grows by 1. ✓
//   ConstrainedStep+AppendConstrainedToken: steps+=1, |generated| grows by 1. ✓
//   RollbackConstrainedSuffix/ToComplete: cost +0, |generated| may shrink (bound preserved). ✓
//   Guards (steps < maxSteps) before each step-consuming call preserve the invariant. ✓
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
        hasSeenOpenSpan := true;
      }
    }
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    var g2, i2, c2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := g2;
    insideConstrainedOut := i2;
    currentConstrainedOut := c2;
    steps := steps + 1;
    spanTokensUsed := 0;
  } else {
    var isDeadEnd := helpers.DeadEndDetection(parser, currentConstrainedOut, 1);
    if isDeadEnd || spanTokensUsed >= spanMaxTokens {
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
      } else if steps < maxSteps {
        // Try one constrained step to reach completion
        var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
        var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
        steps := steps + 1;
        if next == eosToken {
          // Final fallback: use RollbackConstrainedToComplete to guarantee closeable state
          var gFinal, cFinal := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
          generated := gFinal;
          currentConstrainedOut := cFinal;
          if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
            var g3, i3, c3 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
            generated := g3;
            insideConstrainedOut := i3;
            currentConstrainedOut := c3;
            steps := steps + 1;
            spanTokensUsed := 0;
          }
          // If still not complete (empty), continue loop to regenerate
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
            spanTokensUsed := 0;
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
        }
        spanTokensUsed := 0;
        if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
          var g2, i2, c2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
          generated := g2;
          insideConstrainedOut := i2;
          currentConstrainedOut := c2;
          steps := steps + 1;
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
