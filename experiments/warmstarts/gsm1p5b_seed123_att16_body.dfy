    // CSD_RATIONALE_BEGIN
// Analysis of current failures (best: attempt 13 at 44.9%/75.5%):
//
// Key issues from attempt 15 (34.7% accuracy, 57.1% syntax):
// 1. syntax_invalid 19/49: malformed constrained content with {var} curly braces
// 2. unterminated spans: 2 examples opened << but never closed >>
// 3. Mode_C: 8 samples with zero_valid_spans - parser rejecting everything
// 4. Mode_B: 6 samples with partial_valid_spans - some spans valid, some not
//
// Root causes:
// A) The model generates many intermediate spans (avg 3.76/example), most syntactically invalid
//    because they contain reasoning text with {variable} style. Only the LAST span matters.
// B) The approach allows the LM to open constrained spans mid-reasoning, capturing partial
//    expressions. We need to delay opening spans until the LM is ready to emit a complete answer.
// C) The "unterminated" issue: when chunk budget runs out inside a span and rollback yields
//    empty prefix, we need to force-close anyway.
//
// Strategy from attempt 13 (best: 44.9%/75.5%):
// - SafeRepetitionPenaltyStep for constrained generation (not AdaptiveConstrainedStep)
// - chunkSize=10, maxSpanSteps=20, maxSpans=6
// - EnterObservedConstrainedSpan for LM-emitted "<<"
//
// What to improve:
// 1. The main accuracy gap (44.9% -> 41% goal means we're already above the accuracy goal
//    but below syntax goal 90% vs 75.5%). So primary issue is SYNTAX.
//
// 2. To improve syntax: The malformed content comes from the LM emitting {var} tokens inside
//    constrained spans. The SafeRepetitionPenaltyStep uses hard masking, so parser should
//    prevent {var}. BUT the problem is that we're entering constrained spans at the WRONG TIME
//    (too early in reasoning), and the LM generates incomplete/bad expressions.
//
// 3. The "zero_valid_spans" cluster (mode_C: 8 samples) suggests the parser is seeing
//    expressions that never reach a valid complete state. Need to force rollback earlier.
//
// Key change: use AdaptiveConstrainedStep instead of SafeRepetitionPenaltyStep for the
// constrained token selection, with boosting from validTokenGroups. The repetition penalty
// may be penalizing valid tokens in the constrained span.
//
// Also: increase maxSpans to prevent stopping too early when the LM naturally uses
// intermediate spans (some might be valid). But cap visible generation to ensure
// the last span is the answer.
//
// Most important: reduce chunkSize further to 5 tokens so "<<" is detected sooner,
// giving more budget for constrained generation.
//
// For syntax rate improvement:
// - Boost the reserve budget for constrained spans (from 2 to 5)
// - Keep maxSpanSteps=25 to give more room for complete expressions
// - When span fails to close, force exit quickly
//
// Rationale: The best attempt (13) used SafeRepetitionPenalty at 44.9%/75.5%.
// Attempt 15 regressed to 34.7%/57.1% with various changes.
// Going back to attempt 13's core approach but with:
// 1. Smaller chunkSize (5 instead of 10) to detect "<<" faster
// 2. maxSpanSteps=25 (slightly more than 20)
// 3. reserve=3 for span closure (instead of 2)
// 4. maxSpans=8 (more spans allowed)
// 5. Better guidance string
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity (insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)):
//   Initial: established by precondition.
//   EnterObservedConstrainedSpan: sets currentConstrainedOut := [], IsValidPrefix([]) holds.
//   CloseSpanIfComplete (closed=true): insideConstrainedOut := false, implication vacuous.
//   CloseSpanIfComplete (closed=false): state unchanged, invariant preserved.
//   SafeRepetitionPenaltyStep + AppendConstrainedToken: hard mask ensures chosen token extends
//     currentConstrainedOut to a valid prefix; AppendConstrainedToken preserves IsValidPrefix.
//   RollbackConstrainedToComplete: returns complete-or-empty prefix, both satisfy IsValidPrefix.
//     If not complete, we set insideConstrainedOut := false making implication vacuous.
//   CloseConstrainedSpan: sets insideConstrainedOut := false, implication vacuous.
//   OpenConstrainedSpan: sets insideConstrainedOut := true, currentConstrainedOut := []. IsValidPrefix([]) holds.
//
// progress (|generated| <= |generatedPrefix| + steps):
//   UnconstrainedChunk: steps += stepsUsed, |generated| grows by at most stepsUsed. Balanced.
//   EnterObservedConstrainedSpan: +0 steps, generated unchanged. OK.
//   CloseSpanIfComplete (closed=true): steps += 1, generated grows by 1 (">>"). Balanced.
//   SafeRepetitionPenaltyStep + AppendConstrainedToken: steps += 1, generated grows by 1. Balanced.
//   RollbackConstrainedToComplete: +0 steps, generated may shrink. OK.
//   CloseConstrainedSpan: steps += 1, generated grows by 1. Balanced.
//   UnconstrainedStep: steps += 1, generated grows by at most 1. Balanced.
//   Pre-loop first step: steps := 1, generated grows by at most 1. Balanced.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve step by step. Write reasoning in plain text. Place ONLY the final numeric answer inside << >>. Use only: numbers, variable names (letters/underscores only, no curly braces), +, -, *, /, //, **, (, ). One << >> span for the final answer only.");

var steps: nat := 0;
var spanSteps: nat := 0;
var maxSpanSteps: nat := 25;
var chunkSize: nat := 5;
var spansOpened: nat := 0;
var maxSpans: nat := 8;

// Guarantee cost > 0 when maxSteps > 0
if maxSteps > 0 && !insideConstrainedOut {
  var firstNext := helpers.UnconstrainedStep(lm, prompt, generated);
  steps := steps + 1;
  if firstNext == eosToken {
    cost := steps;
    return;
  } else {
    generated := generated + [firstNext];
    if firstNext == "<<" {
      spansOpened := spansOpened + 1;
      var eg0, ei0, ec0 := helpers.EnterObservedConstrainedSpan(lm, generated);
      generated := eg0;
      insideConstrainedOut := ei0;
      currentConstrainedOut := ec0;
      spanSteps := 0;
    }
  }
} else if maxSteps > 0 && insideConstrainedOut {
  var cg0, ci0, cc0, closed0 := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
  steps := steps + 1;
  if closed0 {
    generated := cg0;
    insideConstrainedOut := ci0;
    currentConstrainedOut := cc0;
    spanSteps := 0;
  } else {
    var constrainedPrompt0 := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next0 := helpers.SafeRepetitionPenaltyStep(
      lm, parser, constrainedPrompt0, currentConstrainedOut, generated, 2.0, eosToken
    );
    if next0 == eosToken {
      var rg0, rc0 := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
      generated := rg0;
      currentConstrainedOut := rc0;
      if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
        var closedG0, closedI0, closedC0 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
        generated := closedG0;
        insideConstrainedOut := closedI0;
        currentConstrainedOut := closedC0;
        steps := steps + 1;
        spanSteps := 0;
      } else {
        insideConstrainedOut := false;
        currentConstrainedOut := [];
        spanSteps := 0;
      }
      cost := steps;
      return;
    } else {
      var ag0, ai0, ac0 := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next0
      );
      generated := ag0;
      insideConstrainedOut := ai0;
      currentConstrainedOut := ac0;
      spanSteps := spanSteps + 1;
    }
  }
}

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  if spansOpened >= maxSpans {
    break;
  }

  if !insideConstrainedOut {
    var remaining := maxSteps - steps;
    if remaining < 3 {
      var next1 := helpers.UnconstrainedStep(lm, prompt, generated);
      steps := steps + 1;
      if next1 == eosToken {
        break;
      } else {
        generated := generated + [next1];
        if next1 == "<<" {
          spansOpened := spansOpened + 1;
          var eg1, ei1, ec1 := helpers.EnterObservedConstrainedSpan(lm, generated);
          generated := eg1;
          insideConstrainedOut := ei1;
          currentConstrainedOut := ec1;
          spanSteps := 0;
        }
      }
    } else {
      var budget := if remaining - 2 < chunkSize then remaining - 2 else chunkSize;
      if budget == 0 { budget := 1; }
      var chunkGenerated, stoppedOnOpenSpan, stoppedOnEos, stepsUsed :=
        helpers.UnconstrainedChunk(lm, prompt, generated, budget, "<<", eosToken);
      steps := steps + stepsUsed;
      generated := chunkGenerated;

      if stoppedOnOpenSpan {
        spansOpened := spansOpened + 1;
        var eg2, ei2, ec2 := helpers.EnterObservedConstrainedSpan(lm, generated);
        generated := eg2;
        insideConstrainedOut := ei2;
        currentConstrainedOut := ec2;
        spanSteps := 0;
      } else if stoppedOnEos {
        break;
      }
    }
  } else {
    var remaining3 := maxSteps - steps;
    var shouldForceClose := spanSteps >= maxSpanSteps || remaining3 <= 1;

    if shouldForceClose {
      var rg3, rc3 := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
      generated := rg3;
      currentConstrainedOut := rc3;
      if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
        var closedG3, closedI3, closedC3 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
        generated := closedG3;
        insideConstrainedOut := closedI3;
        currentConstrainedOut := closedC3;
        steps := steps + 1;
        spanSteps := 0;
      } else {
        insideConstrainedOut := false;
        currentConstrainedOut := [];
        spanSteps := 0;
        if steps < maxSteps {
          var dummy := helpers.UnconstrainedStep(lm, prompt, generated);
          steps := steps + 1;
          if dummy == eosToken {
            break;
          } else {
            generated := generated + [dummy];
            if dummy == "<<" {
              spansOpened := spansOpened + 1;
              var eg4, ei4, ec4 := helpers.EnterObservedConstrainedSpan(lm, generated);
              generated := eg4;
              insideConstrainedOut := ei4;
              currentConstrainedOut := ec4;
              spanSteps := 0;
            }
          }
        }
      }
    } else {
      var cg5, ci5, cc5, closed5 := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
      if closed5 {
        steps := steps + 1;
        generated := cg5;
        insideConstrainedOut := ci5;
        currentConstrainedOut := cc5;
        spanSteps := 0;
      } else {
        var constrainedPrompt5 := prompt + generated[..|generated| - |currentConstrainedOut|];
        var next5 := helpers.SafeRepetitionPenaltyStep(
          lm, parser, constrainedPrompt5, currentConstrainedOut, generated, 2.0, eosToken
        );
        steps := steps + 1;
        spanSteps := spanSteps + 1;
        if next5 == eosToken {
          var rg5, rc5 := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
          generated := rg5;
          currentConstrainedOut := rc5;
          if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
            var closedG5, closedI5, closedC5 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
            generated := closedG5;
            insideConstrainedOut := closedI5;
            currentConstrainedOut := closedC5;
            steps := steps + 1;
            spanSteps := 0;
          } else {
            insideConstrainedOut := false;
            currentConstrainedOut := [];
            spanSteps := 0;
          }
          break;
        } else {
          var ag5, ai5, ac5 := helpers.AppendConstrainedToken(
            lm, parser, generated, currentConstrainedOut, next5
          );
          generated := ag5;
          insideConstrainedOut := ai5;
          currentConstrainedOut := ac5;
        }
      }
    }
  }
}

// Final cleanup: if still inside a constrained span, close it
if insideConstrainedOut {
  var rg6, rc6 := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
  generated := rg6;
  currentConstrainedOut := rc6;
  if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
    var closedG6, closedI6, closedC6 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := closedG6;
    insideConstrainedOut := closedI6;
    currentConstrainedOut := closedC6;
    steps := steps + 1;
  } else {
    insideConstrainedOut := false;
    currentConstrainedOut := [];
  }
}

cost := steps;
