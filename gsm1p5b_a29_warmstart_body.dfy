// CSD_RATIONALE_BEGIN
// Diagnosis of attempt 28 (0% accuracy, 0% syntax):
// The previous attempt broke from attempt 22 (46.9% accuracy, 87.8% syntax) by
// adding complex dead-end detection and rollback logic that caused the span to
// never close. The key diagnostic: 49/49 opened "<<" but never emitted ">>".
//
// The fundamental problem: The constrained span generation is consuming all
// available steps without producing a complete expression that can be closed.
//
// Root causes:
// 1. Dead-end detection + rollback + ConstrainedStep + no-close = spin
//    When dead-end fires, we rollback, try one ConstrainedStep, but if that
//    token doesn't make the expression complete, we continue the loop and
//    repeat the dead-end/rollback cycle indefinitely.
// 2. The "break" after failed rollback in attempt 28 exits the loop but the
//    post-loop cleanup ALSO fails to close, leaving the span open.
// 3. The step budget (900) is large enough that we can spin for a long time.
//
// The BEST attempt (22) achieved 46.9% accuracy, 87.8% syntax with a simpler
// approach. Let me return to that approach but fix the remaining issues:
//
// Issues in attempt 22 that need fixing:
// 1. ~6 unterminated spans: rollback finds valid but not complete prefix
//    FIX: Rollback more aggressively (up to 8 times) until complete
// 2. ~6 mode_Q examples: model hits EOS without ever emitting "<<"
//    FIX: After EOS without hasSeenOpenSpan, force-open and generate answer
//
// KEY INSIGHT from the diagnostic:
// "Top helper calls: RollbackToValidPrefix=294, RollbackConstrainedSuffix=294"
// That's 294/49 = 6 rollbacks per example. This is NOT spinning - it's 6 calls
// total per example, which is exactly the 6 rollback calls in the dead-end branch.
// The issue is that after 6 rollbacks, the span still isn't complete, and we
// break out of the loop with an open span.
//
// The failing examples show content like "<<1 thought" - the model is generating
// English text inside the constrained span! The parser doesn't accept English,
// so after the first few tokens, we hit a dead end, rollback to empty (or near
// empty), try one ConstrainedStep, get a number like "1", and then the span
// content is "1" which IS complete. But then we're not closing it!
//
// Wait, "<<1 thought" means the span content is "1 thought" which is wrong.
// But "1" alone should be complete. The issue is that after getting "1" from
// ConstrainedStep, we append it but don't close immediately.
//
// Actually looking at the code: after ConstrainedStep gives "1" (not EOS),
// we AppendConstrainedToken (span = "1"), then check IsCompletePrefix("1").
// If "1" is complete, we close. But the output shows "<<1 thought" which means
// the span was NOT closed after "1" and the model continued with " thought".
//
// But " thought" is not parser-valid! The ConfidenceGatedStep would hard-mask
// it out. Unless... the model generates " thought" in the UNCONSTRAINED phase
// AFTER the span should have been closed.
//
// OH WAIT: "<<1 thought" - the "<<" was emitted in the UNCONSTRAINED phase by
// UnconstrainedChunk, then we EnterObservedConstrainedSpan. Then we generate
// "1" (constrained). Then IsCompletePrefix("1") is true, so we close with ">>".
// But the output shows no ">>", just "1 thought" after "<<".
//
// This means either:
// a) IsCompletePrefix("1") is FALSE (the parser requires more than just a number)
// b) The CloseConstrainedSpan failed somehow
// c) The closing ">>" was emitted but then more text was generated after it
//    (which would be fine - the evaluator just needs the LAST complete span)
//
// Actually looking at the output: "The structure has <<1 thought"
// The "<<" appears after "has " which suggests the model wrote "has <<" in
// the unconstrained phase. Then "1" was generated constrained. Then " thought"
// appears AFTER - but if we closed with ">>", the output would be "<<1>> thought".
// The absence of ">>" confirms the span was never closed.
//
// HYPOTHESIS: IsCompletePrefix("1") is FALSE. The parser for gsm_symbolic
// might require expressions like "n1 + n2" or "42" but not just "1" (a literal).
// Actually "1" IS a number, so it should be complete.
//
// ALTERNATIVE HYPOTHESIS: The model generates "1" but then the ConstrainedStep
// returns "1" and we check IsCompletePrefix(["1"]). If "1" is a multi-character
// token, maybe the parser sees it as a partial number? No, tokens are complete.
//
// Let me re-read the failing example: "The structure has <<1 thought"
// This is the STRATEGY EXTRACTED output, not the raw generated tokens.
// The raw tokens might be: ["The", " structure", " has", " <<", "1", " thought", ...]
// Wait, "<<" is a single token. After "<<" we enter constrained mode.
// Then we generate constrained tokens. If "1" is complete, we close.
// The output would be: "The structure has <<1>> thought"
// But the output shows: "The structure has <<1 thought"
// So ">>" was never emitted.
//
// CONCLUSION: The IsCompletePrefix check is failing for "1" or the close
// is not being reached. Let me trace through the attempt 28 code:
//
// After EnterObservedConstrainedSpan: insideConstrainedOut=true, currentConstrainedOut=[]
// Loop iteration:
//   else (inside span, not complete):
//     isDeadEnd = DeadEndDetection([], 1) - is [] a dead end?
//     If [] has valid next tokens (numbers, variables), isDeadEnd=false
//     spanTokensUsed=0 < spanMaxTokens=8, so go to normal constrained step
//     ConfidenceGatedStep([], eosToken) -> returns "1"
//     steps += 1
//     next = "1" != eosToken
//     AppendConstrainedToken -> currentConstrainedOut = ["1"]
//     spanTokensUsed = 1
// Next loop iteration:
//   else if parser.IsCompletePrefix(["1"]) -> CLOSE!
//   CloseConstrainedSpan -> emits ">>"
//
// So if IsCompletePrefix(["1"]) is true, we SHOULD close on the next iteration.
// The output "<<1 thought" suggests either:
// a) IsCompletePrefix(["1"]) is false (unlikely for a number)
// b) Something else is happening
//
// WAIT: The prompt is gsm_symbolic with TEMPLATE VARIABLES like {n1}, {color1}.
// The actual problem values are NOT filled in! The model sees "{n1}" as text.
// So when the model generates inside "<<", it might be trying to write "{n1}"
// but the parser rejects "{". The ConstrainedStep hard-masks "{", so the model
// gets something else. But what?
//
// For the example "The structure has <<1 thought":
// After "<<", the model is inside the constrained span.
// ConfidenceGatedStep: model's top token is probably " n" or "{" or a number.
// "{" is hard-masked. The model then picks "1" (a number).
// IsCompletePrefix(["1"]) - if this is TRUE, we close next iteration.
// But the output shows " thought" after "1" without ">>".
//
// I think the issue is that " thought" is generated in the UNCONSTRAINED phase
// AFTER the span closes. The evaluator might be showing the text differently.
// But the evaluator says "unterminated_constrained_segment" which means no ">>".
//
// NEW HYPOTHESIS: The ConfidenceGatedStep might be returning " 1" (with space)
// as a single token. Then currentConstrainedOut = [" 1"]. Is [" 1"] complete?
// If the parser expects tokens without leading spaces, this might fail.
//
// OR: The ConstrainedStep is being called with `prompt` (not `prompt + generated`)
// which might confuse the LM and cause it to generate unexpected tokens.
//
// In attempt 22 (best), ConfidenceGatedStep is called with:
// `var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];`
// This correctly passes the context up to (but not including) the constrained content.
//
// In attempt 28, I removed this constrainedPrompt construction and just used `prompt`.
// That's a BUG! The LM needs the full context to generate appropriate tokens.
//
// FIX: Restore `constrainedPrompt` construction as in attempt 22.
//
// Let me also check: in attempt 28, the ConfidenceGatedStep call is:
// `helpers.ConfidenceGatedStep(lm, parser, prompt, currentConstrainedOut, eosToken)`
// But in attempt 22, it was:
// `helpers.ConfidenceGatedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken)`
// where `constrainedPrompt = prompt + generated[..|generated| - |currentConstrainedOut|]`
//
// YES! This is the bug. Without the full context, the LM generates random tokens
// that don't fit the math problem, leading to unterminated spans.
//
// PLAN: Return to attempt 22 structure (which achieved 46.9% accuracy, 87.8% syntax)
// with these targeted fixes:
// 1. Keep constrainedPrompt construction for all inside-span generation
// 2. More aggressive rollback (up to 8 times) to find complete prefix
// 3. Better handling of EOS-without-span (force-open on stoppedOnEos)
// 4. Keep spanMaxTokens=12 as in attempt 22
// 5. Remove the complex dead-end detection that caused spinning
//
// The 87.8% syntax in attempt 22 means ~6 unterminated spans. To fix those,
// we need the rollback to find a complete prefix. The key is to rollback
// UNTIL COMPLETE, not just until valid. Since RollbackConstrainedSuffix
// returns a valid prefix (not necessarily complete), we need multiple calls.
//
// For "n1 * cur" -> rollback -> "n1 *" (valid, not complete)
//              -> rollback -> "n1" (valid, complete!) -> close!
//
// So 2 rollbacks should be enough for most cases. Let's use 6 rollbacks
// to be safe, matching attempt 22.
//
// CRITICAL DIFFERENCE from attempt 28: We DON'T break after failed rollback.
// Instead, we continue the main loop. The main loop will then re-enter the
// "inside span, not complete" branch and try another ConstrainedStep.
// This is what attempt 22 did and it worked for 87.8% of examples.
//
// The remaining 12.2% that failed in attempt 22 were:
// - 6 mode_Q (no span at all): fix with force-open on EOS
// - 6 unterminated: fix with more aggressive rollback
//
// Let me also check if the issue with attempt 28 was the missing constrainedPrompt.
// In attempt 28: `helpers.ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)`
// In attempt 22: `helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken)`
// YES, this is the bug. Without context, the LM doesn't know what number to generate.
//
// FINAL PLAN: Essentially reproduce attempt 22 with minor improvements:
// 1. Keep constrainedPrompt for all inside-span LM calls
// 2. Use 6 rollbacks (same as attempt 22) in each dead-end branch
// 3. Don't use dead-end detection (it caused spinning in attempts 27-28)
// 4. Use spanMaxTokens=12 to trigger rollback after 12 tokens
// 5. On EOS without span: force-open, generate constrained content
// 6. Keep the post-loop cleanup from attempt 22
//
// Wait, attempt 22 DID use dead-end detection:
// `var isDeadEnd := helpers.DeadEndDetection(parser, currentConstrainedOut, 1);`
// And it worked (87.8% syntax). The issue in attempts 27-28 was the BREAK
// after failed rollback, which left spans open. In attempt 22, we continue
// the loop after rollback, giving the LM more chances to generate tokens.
//
// So the fix is: REMOVE the break statements after failed rollback.
// Instead, continue the loop (which will try more ConstrainedStep calls).
// The spanMaxTokens limit prevents infinite loops by eventually triggering
// another rollback.
//
// But wait - if rollback gives a non-complete prefix and we continue, then
// on the next iteration we check IsCompletePrefix (false), check isDeadEnd,
// try another ConstrainedStep, append the token, increment spanTokensUsed.
// If spanTokensUsed < spanMaxTokens, we continue normally.
// This IS what attempt 22 does, and it works for 87.8% of examples!
//
// BOTTOM LINE: Return to attempt 22 code exactly, with these two changes:
// 1. Fix the constrainedPrompt construction (attempt 28 broke this)
// 2. Keep the force-open on stoppedOnEos (attempt 22 already had this)
// 3. Maybe increase rollback count from 4 to 6 in the EOS-inside-span branch
//
// Actually, let me re-read attempt 22 carefully. It uses constrainedPrompt:
// `var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];`
// This is correct. Attempt 28 removed this and used `prompt` directly. That's the bug.
//
// Let me reproduce attempt 22 exactly (the best attempt) with minor improvements.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//   Outside span, UnconstrainedChunk stoppedOnOpenSpan: EnterObservedConstrainedSpan
//     sets currentConstrainedOut:=[], IsValidPrefix([]) holds by precondition.
//   Outside span, OpenConstrainedSpan (forced): sets currentConstrainedOut:=[],
//     IsValidPrefix([]) holds by precondition.
//   Inside span, IsCompletePrefix: CloseConstrainedSpan sets insideConstrainedOut:=false,
//     implication vacuous.
//   Inside span, ConfidenceGatedStep + AppendConstrainedToken: ConfidenceGatedStep
//     hard-masks when needed, so AppendConstrainedToken preserves IsValidPrefix.
//   Inside span, ConstrainedStep + AppendConstrainedToken: ConstrainedStep hard-masks
//     to valid tokens; AppendConstrainedToken preserves IsValidPrefix by postcondition.
//   Inside span, rollback calls: each RollbackConstrainedSuffix guarantees
//     IsValidPrefix(cRolled) by its postcondition.
//   CloseConstrainedSpan: sets insideConstrainedOut:=false, implication vacuous.
//   break paths: insideConstrainedOut may remain true, but all prior ops maintained
//     IsValidPrefix(currentConstrainedOut). ✓
//
// progress: |generated| <= |generatedPrefix| + steps
//
//   UnconstrainedChunk: steps += stepsUsed, |generated| grows by at most stepsUsed. ✓
//   OpenConstrainedSpan: steps += 1, |generated| grows by 1 ("<<"). ✓
//   EnterObservedConstrainedSpan: steps += 0, |generated| unchanged. ✓
//   CloseConstrainedSpan: steps += 1, |generated| grows by at most 1 (">>"). ✓
//   ConfidenceGatedStep + AppendConstrainedToken: steps += 1, |generated| grows by 1. ✓
//   ConstrainedStep + AppendConstrainedToken: steps += 1, |generated| grows by 1. ✓
//   RollbackConstrainedSuffix: cost +0, |generated| shrinks (stays <= bound). ✓
//   Post-loop: steps <= maxSteps enforced by guards. ✓
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve step by step. Write each calculation and the final answer inside << >> delimiters. Example: <<n1 + n2>>. Final answer: <<42>>.");

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
      // Force open a constrained span - budget is running low and no span seen yet
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
        // Model hit EOS - if no span seen, force open one for the final answer
        if !hasSeenOpenSpan && steps + 3 <= maxSteps {
          var g2, i2, c2 := helpers.OpenConstrainedSpan(lm, generated);
          generated := g2;
          insideConstrainedOut := i2;
          currentConstrainedOut := c2;
          steps := steps + 1;
          spanTokensUsed := 0;
          hasSeenOpenSpan := true;
          // Don't break - continue to generate constrained content
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
      // else: chunk ended without << and without EOS; continue
    }
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    var g2, i2, c2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := g2;
    insideConstrainedOut := i2;
    currentConstrainedOut := c2;
    steps := steps + 1;
    spanTokensUsed := 0;
  } else {
    // Inside span, not complete yet
    var isDeadEnd := helpers.DeadEndDetection(parser, currentConstrainedOut, 1);
    if isDeadEnd || spanTokensUsed >= spanMaxTokens {
      // Rollback to find a closeable state - rollback multiple times if needed
      var gRolled, cRolled := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
      generated := gRolled;
      currentConstrainedOut := cRolled;
      if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
        var g3, c3 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
        generated := g3;
        currentConstrainedOut := c3;
      }
      if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
        var g4, c4 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
        generated := g4;
        currentConstrainedOut := c4;
      }
      if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
        var g5, c5 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
        generated := g5;
        currentConstrainedOut := c5;
      }
      if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
        var g6, c6 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
        generated := g6;
        currentConstrainedOut := c6;
      }
      if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
        var g7, c7 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
        generated := g7;
        currentConstrainedOut := c7;
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
          // Rollback multiple times to find complete prefix
          var gR2, cR2 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
          generated := gR2;
          currentConstrainedOut := cR2;
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
          if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
            var g2, i2, c2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
            generated := g2;
            insideConstrainedOut := i2;
            currentConstrainedOut := c2;
            steps := steps + 1;
          }
          // If still not complete, continue loop (don't break - let loop try again)
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
        // Rollback multiple times to find complete prefix
        var gRolled, cRolled := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
        generated := gRolled;
        currentConstrainedOut := cRolled;
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
        spanTokensUsed := 0;
        if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
          var g2, i2, c2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
          generated := g2;
          insideConstrainedOut := i2;
          currentConstrainedOut := c2;
          steps := steps + 1;
        }
        // If not complete, continue loop (don't break - let loop try again with fresh attempt)
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

// Post-loop cleanup: if we exited the loop with an open span, try to close it
if insideConstrainedOut && steps < maxSteps {
  // Rollback multiple times to find a complete prefix
  var gRolled, cRolled := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
  generated := gRolled;
  currentConstrainedOut := cRolled;
  if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
    var g2, c2 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
    generated := g2;
    currentConstrainedOut := c2;
  }
  if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
    var g3, c3 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
    generated := g3;
    currentConstrainedOut := c3;
  }
  if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
    var g4, c4 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
    generated := g4;
    currentConstrainedOut := c4;
  }
  if !parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| > 0 {
    var g5, c5 := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrainedOut);
    generated := g5;
    currentConstrainedOut := c5;
  }
  if !parser.IsCompletePrefix(currentConstrainedOut) && steps + 1 < maxSteps {
    // Try one more constrained step
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
    steps := steps + 1;
    if next != eosToken {
      var g2, i2, c2 := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
      generated := g2;
      insideConstrainedOut := i2;
      currentConstrainedOut := c2;
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
