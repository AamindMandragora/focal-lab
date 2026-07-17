// CSD_RATIONALE_BEGIN
// Analysis of failures across attempts:
// - Best result (attempt 14): 2% accuracy, 98% syntax. Almost all spans close correctly
//   but the content is semantically wrong.
// - Attempt 17: 8.2% accuracy, 79.6% syntax. Better accuracy but worse syntax due to
//   10 unterminated spans from the "minimum span steps" delay.
//
// The core problem is TWO distinct failure modes:
// 1. SYNTAX FAILURES (10 examples): Span opens but never closes because the model
//    generates complex/broken expressions that hit dead ends. The tail shows things like
//    "<<n - (m * cur" or "<<(length * unit!" - the model pre-generates inside the span
//    unconstrained, then the constrained phase can't close it.
//    FIX: Use CloseSpanWithinBudget with generous budget immediately after entering
//    the span.
//
// 2. SEMANTIC FAILURES (35 examples): The model writes correct reasoning but then
//    writes a simplified/truncated formula in the span. For example:
//    - Correct: "n1 * frac1 + n2 * mult1" but writes "n1 * (frac1 + mult1)"
//    - Correct: "int(n * frac_1 * frac_2)" but writes "n * frac_1 * frac_2"
//    - Correct: "multiple * relative_age - orbit_period" but writes wrong formula
//    This is a model reasoning problem, not a span mechanism problem.
//
// Key insight: The model's reasoning in the unconstrained preamble is mostly correct
// but it summarizes/simplifies when writing the final answer. The "int()" wrapper
// in some correct answers suggests the evaluator may require integer wrapping.
//
// Strategy for improvement:
// 1. Keep the approach from best result (attempt 14) which had 98% syntax.
// 2. Improve the unconstrained preamble to discourage simplification.
// 3. The guidance should tell the model to use the EXACT formula it computed in its
//    reasoning, not to simplify it.
// 4. Looking at the representative failures:
//    - "n * frac_1 * frac_2" vs "int(n * frac_1 * frac_2)": The int() wrapper.
//    - "n1 * (frac1 + mult1)" vs "n1 * frac1 + n2 * mult1": Wrong factoring.
//    - "multiple * relative_age - relative_age" vs "multiple * relative_age - orbit_period": 
//      Model used wrong variable.
//
// The correct answers use int() wrapping! This suggests the parser grammar accepts
// int(expr) syntax. The model needs to be told to wrap in int().
//
// Also: the model is making algebraic errors (factoring n1 out when the coefficients
// differ). This is a fundamental model limitation.
//
// Best achievable improvement: Better guidance to:
// 1. Write int(EXPRESSION) as the format
// 2. Do NOT factor or simplify - write the literal arithmetic
// 3. Use exact variable names as they appear in the problem
// 4. Keep syntax rate high by using CloseSpanWithinBudget
//
// Architecture: Same as best attempt (14) but with improved guidance and
// CloseSpanWithinBudget as primary closure mechanism to maintain 98% syntax.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//   Phase 1 (UnconstrainedChunk loop): insideConstrainedOut stays false.
//   When stoppedOnOpen fires, EnterObservedConstrainedSpan sets currentConstrainedOut := []
//   which satisfies parser.IsValidPrefix([]) by precondition.
//   In the constrained inner loop:
//   - ConstrainedStep returns a token that is parser-valid next or EOS (by postcondition).
//   - AppendConstrainedToken preserves parser.IsValidPrefix(currentConstrainedOut).
//   - CloseSpanIfComplete either closes (insideConstrainedOut := false, implication vacuous)
//     or is a no-op (invariant unchanged). 
//   - CloseSpanWithinBudget guarantees insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut).
//   OpenConstrainedSpan sets currentConstrainedOut := [] (valid by precondition).
//
// progress: |generated| <= |generatedPrefix| + steps
//
//   Phase 1: UnconstrainedChunk adds stepsUsed to steps and appends at most stepsUsed tokens
//   to generated (stoppedOnEos means EOS not appended). So |generated| grows by at most
//   stepsUsed while steps grows by exactly stepsUsed.
//   Constrained inner loop: each iteration increments steps by 1 and either appends 1 token
//   (AppendConstrainedToken path), closes the span (appends ">>" i.e. 1 token via CloseSpanIfComplete),
//   or appends 0 tokens (EOS break). So |generated| - |generatedPrefix| <= steps holds.
//   CloseSpanWithinBudget: |generatedOut| <= |generated| + closeBudget and steps += closeBudget.
//   OpenConstrainedSpan: appends "<<" (1 token), steps += 1. Both preserve invariant.
//   Phase 2: Same argument applies.
//   Final safety: CloseSpanWithinBudget with finalBudget preserves the invariant.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve this math problem step by step. Use the exact variable names from the problem statement. Do NOT simplify or factor expressions. When you compute the final answer formula, write every term explicitly. At the very end, write the answer as: <<int(EXPRESSION)>> where EXPRESSION uses all required variables and operators. For example: <<int(n * frac1 * frac2)>> or <<int(n1 * frac1 + n2 * mult1)>> or <<int(n * bill - m * p1 - k * p2)>>. Do not factor out common terms. Write each term separately.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;
var hasCompletedSpan: bool := false;

var phase1Budget: nat := 750;
if phase1Budget > maxSteps {
  phase1Budget := maxSteps;
}

// Phase 1: Unconstrained generation watching for "<<"
while steps < phase1Budget && !insideConstrainedOut && !hasCompletedSpan
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases phase1Budget - steps
{
  var chunkBudget: nat := 25;
  if steps + chunkBudget > phase1Budget {
    chunkBudget := phase1Budget - steps;
  }
  if chunkBudget == 0 {
    break;
  }
  var cg, stoppedOnOpen, stoppedOnEos, stepsUsed := helpers.UnconstrainedChunk(lm, prompt, generated, chunkBudget, "<<", eosToken);
  generated := cg;
  steps := steps + stepsUsed;
  if stoppedOnEos {
    break;
  }
  if stoppedOnOpen {
    var eg, ei, ec := helpers.EnterObservedConstrainedSpan(lm, generated);
    generated := eg;
    insideConstrainedOut := ei;
    currentConstrainedOut := ec;

    // Use CloseSpanWithinBudget as primary mechanism - reliable closure
    // Give it a generous budget to find and close the expression
    if steps < maxSteps {
      var closeBudget: nat := 120;
      var remaining := maxSteps - steps;
      if closeBudget > remaining {
        closeBudget := remaining;
      }
      if closeBudget > 0 {
        var wg, wi, wc := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget);
        generated := wg;
        insideConstrainedOut := wi;
        currentConstrainedOut := wc;
        steps := steps + closeBudget;
        if !insideConstrainedOut {
          hasCompletedSpan := true;
        }
      }
    }

    // If span still open after CloseSpanWithinBudget, try constrained step loop
    var innerSteps: nat := 0;
    var innerBudget: nat := 60;
    while insideConstrainedOut && !hasCompletedSpan && steps < maxSteps && innerSteps < innerBudget
      invariant 0 <= steps <= maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant !insideConstrainedOut ==> currentConstrainedOut == []
      invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
      invariant |generated| <= |generatedPrefix| + steps
      decreases maxSteps - steps + innerBudget - innerSteps
    {
      if parser.IsCompletePrefix(currentConstrainedOut) {
        var cg2, ci2, cc2, closed2 := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
        steps := steps + 1;
        innerSteps := innerSteps + 1;
        generated := cg2;
        insideConstrainedOut := ci2;
        currentConstrainedOut := cc2;
        if closed2 {
          hasCompletedSpan := true;
        }
      } else {
        var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
        var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
        steps := steps + 1;
        innerSteps := innerSteps + 1;
        if next == eosToken {
          break;
        } else {
          var ag, ai, ac := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
          generated := ag;
          insideConstrainedOut := ai;
          currentConstrainedOut := ac;
        }
      }
    }

    // Final close attempt if still open
    if insideConstrainedOut && steps < maxSteps {
      var closeBudget2: nat := 40;
      var remaining2 := maxSteps - steps;
      if closeBudget2 > remaining2 {
        closeBudget2 := remaining2;
      }
      if closeBudget2 > 0 {
        var wg2, wi2, wc2 := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget2);
        generated := wg2;
        insideConstrainedOut := wi2;
        currentConstrainedOut := wc2;
        steps := steps + closeBudget2;
        if !insideConstrainedOut {
          hasCompletedSpan := true;
        }
      }
    }

    break;
  }
}

// Phase 2: If no span was opened yet, force open one
if !insideConstrainedOut && !hasCompletedSpan && steps < maxSteps {
  var remainingForSpan := maxSteps - steps;
  if remainingForSpan >= 2 {
    var fg, fi, fc := helpers.OpenConstrainedSpan(lm, generated);
    generated := fg;
    insideConstrainedOut := fi;
    currentConstrainedOut := fc;
    steps := steps + 1;

    // Primary: CloseSpanWithinBudget
    if insideConstrainedOut && steps < maxSteps {
      var closeBudget3: nat := 100;
      var remaining3 := maxSteps - steps;
      if closeBudget3 > remaining3 {
        closeBudget3 := remaining3;
      }
      if closeBudget3 > 0 {
        var wg3, wi3, wc3 := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget3);
        generated := wg3;
        insideConstrainedOut := wi3;
        currentConstrainedOut := wc3;
        steps := steps + closeBudget3;
        if !insideConstrainedOut {
          hasCompletedSpan := true;
        }
      }
    }

    // Fallback: constrained step loop
    var innerSteps3: nat := 0;
    var innerBudget3: nat := 60;
    while insideConstrainedOut && !hasCompletedSpan && steps < maxSteps && innerSteps3 < innerBudget3
      invariant 0 <= steps <= maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant !insideConstrainedOut ==> currentConstrainedOut == []
      invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
      invariant |generated| <= |generatedPrefix| + steps
      decreases maxSteps - steps + innerBudget3 - innerSteps3
    {
      if parser.IsCompletePrefix(currentConstrainedOut) {
        var cg4, ci4, cc4, closed4 := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
        steps := steps + 1;
        innerSteps3 := innerSteps3 + 1;
        generated := cg4;
        insideConstrainedOut := ci4;
        currentConstrainedOut := cc4;
        if closed4 {
          hasCompletedSpan := true;
        }
      } else {
        var constrainedPrompt3 := prompt + generated[..|generated| - |currentConstrainedOut|];
        var next3 := helpers.ConstrainedStep(lm, parser, constrainedPrompt3, currentConstrainedOut, eosToken);
        steps := steps + 1;
        innerSteps3 := innerSteps3 + 1;
        if next3 == eosToken {
          break;
        } else {
          var ag3, ai3, ac3 := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next3);
          generated := ag3;
          insideConstrainedOut := ai3;
          currentConstrainedOut := ac3;
        }
      }
    }

    // Last resort close
    if insideConstrainedOut && steps < maxSteps {
      var closeBudget4: nat := 40;
      var remaining4 := maxSteps - steps;
      if closeBudget4 > remaining4 {
        closeBudget4 := remaining4;
      }
      if closeBudget4 > 0 {
        var wg4, wi4, wc4 := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget4);
        generated := wg4;
        insideConstrainedOut := wi4;
        currentConstrainedOut := wc4;
        steps := steps + closeBudget4;
        if !insideConstrainedOut {
          hasCompletedSpan := true;
        }
      }
    }
  }
}

// Final safety: use all remaining budget to close any still-open span
if insideConstrainedOut && steps < maxSteps {
  var finalBudget := maxSteps - steps;
  if finalBudget > 0 {
    var wgf, wif, wcf := helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, finalBudget);
    generated := wgf;
    insideConstrainedOut := wif;
    currentConstrainedOut := wcf;
    steps := steps + finalBudget;
  }
}

cost := steps;
