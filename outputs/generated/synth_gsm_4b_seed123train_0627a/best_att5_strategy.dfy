// CSD_RATIONALE_BEGIN
// Analysis of all 4 attempts:
// - Best result: attempt 2 (63.3% acc, 71.4% syntax) used CloseSpanWithinBudget for entire inside-span phase
// - Attempt 4 (44.9% acc, 71.4% syntax) used hybrid ConstrainedStep + CloseSpanWithinBudget - regressed
// - Primary gaps: syntax 71.4% vs goal 93%, accuracy 63.3% vs goal 49%
//
// Key failure modes from attempt 4:
// 1. 7/49 span-unclosed (CloseSpanWithinBudget wasn't being reached or budget too small)
// 2. 6/49 final_span_invalid (malformed content: {}, ** etc.)
// 3. 6/49 no_span_emitted (model never generates <<)
// 4. 13/49 syntax_valid_wrong (semantic errors - model picks wrong formula)
//
// Critical observation: "Constrained intervention activity: examples_with_activity 4/49"
// This means 45/49 examples had NO constrained steps - the model generated << in unconstrained
// mode, then CloseSpanWithinBudget was doing all the span work. Yet syntax was only 71.4%.
//
// The CloseSpanWithinBudget approach in attempt 2 achieved 71.4% syntax. The problem is
// that CloseSpanWithinBudget uses free generation internally for content, which can include
// curly braces and other invalid tokens. The parser should be preventing this.
//
// New diagnosis: The parser IS being used in CloseSpanWithinBudget (it's constrained), but
// the failures are in CONTENT that's syntactically wrong (curly braces are somehow passing
// the parser) - OR the model is generating valid symbolic expressions that just have wrong
// variable names (semantic failures).
//
// Key insight from feedback: "Unit-rewind opportunity: most failing examples produced 
// well-formed output that passed the syntax check but scored incorrect"
// This suggests RegenerateUnitOnGroundingFailure could help for semantic accuracy.
//
// Strategy redesign:
// 1. Keep unconstrained free generation outside spans (model is good at this)
// 2. When << detected in free generation, enter constrained mode
// 3. Inside spans: use ConstrainedGeneration or CloseSpanWithinBudget with a budget
//    that's generous enough to complete the span
// 4. Key fix: don't use hybrid with ConstrainedStep first - just use CloseSpanWithinBudget
//    with ALL remaining steps (like attempt 2) which was the BEST result
// 5. BUT also add RepetitionPenaltyStep / penalization to help avoid curly braces
//
// The regression from attempt 2->4 was caused by the ConstrainedStep hybrid loop.
// Go back to the attempt 2 pattern but add:
// - Better guidance text
// - Explicit handling for budget exhaustion to avoid the 7 unclosed spans
// - CloseSpanWithinBudget with the full remaining budget immediately on span entry
//
// The 6 no_span_emitted cases: these are model behavior cases where the model reasons
// at length without using <<. We can try to guide the model to use << more consistently.
// But fundamentally this is hard to fix with CSD alone.
//
// The main opportunity: attempt 2 had syntax 71.4%. We need to get to 93%.
// The 14 syntax failures in attempt 4 broke down as:
// - 2 final_span_unclosed
// - 6 no_span_emitted
// - 6 final_span_invalid (content has {} ** etc.)
//
// The 6 no_span_emitted cases are the hardest. We need the model to emit <<.
// One approach: use UnconstrainedChunk which can stop when << is observed.
// Then when stoppedOnOpenSpan, use EnterObservedConstrainedSpan + CloseSpanWithinBudget.
// This is the same pattern but lets us observe << more reliably.
//
// For final_span_invalid: the parser SHOULD prevent {}, ** etc. If these are appearing,
// it means either:
// (a) The spans were closed BEFORE entering constrained mode (free gen produced content)
// (b) The parser allows {} (unlikely - parser should be a symbolic expression grammar)
// Looking at the output tails: "<<{n1} * {w1} + {n2} * {w2} + {n3} * {w3}>>"
// These curly braces ARE appearing inside the << >> delimiters in the FINAL output.
// This means the model generated them in FREE (unconstrained) mode before the << was
// detected by our strategy. The strategy detects << only after UnconstrainedStep returns "<<".
// But the model is generating COMPLETE spans like "<<expr>>" in a single unconstrained chunk!
//
// CRITICAL FIX: The model is generating "<<...>>" ALL in unconstrained mode because our
// strategy only detects << when UnconstrainedStep RETURNS "<<". But by then, the model
// has ALREADY generated the content including ">>". The content was generated freely.
//
// Solution: Use UnconstrainedChunk to stop AT the << token, then use CloseSpanWithinBudget
// to generate the content under parser control. This ensures the content is constrained.
//
// Implementation plan:
// 1. Use UnconstrainedChunk (stops at "<<", stoppedOnOpenSpan=true when << detected)
// 2. When stoppedOnOpenSpan: use EnterObservedConstrainedSpan, then CloseSpanWithinBudget
// 3. When !stoppedOnOpenSpan: we hit EOS or budget limit
// 4. This ensures span CONTENT is always parser-controlled
//
// Budget management: maxSteps=900, stepTokenBudget=1
// Average generated tokens: 481, median 321
// With UnconstrainedChunk doing the free generation and CloseSpanWithinBudget doing spans,
// we should be able to handle most cases within budget.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
// Main loop invariant preservation:
//
// 1. parser_validity:
//    - !insideConstrainedOut branch (UnconstrainedChunk):
//      When stoppedOnOpenSpan is true, EnterObservedConstrainedSpan sets
//      insideConstrainedOut := true and currentConstrainedOut := []. Since
//      parser.IsValidPrefix([]) holds by precondition, the invariant is satisfied.
//      When !stoppedOnOpenSpan, insideConstrainedOut remains false, so the implication
//      is vacuous.
//    - insideConstrainedOut branch (CloseSpanWithinBudget):
//      CloseSpanWithinBudget's postcondition guarantees either !insideConstrainedOut
//      (making the implication vacuous, with currentConstrainedOut == []) or
//      parser.IsValidPrefix(currentConstrainedOut). So the invariant is preserved.
//
// 2. progress: |generated| <= |generatedPrefix| + steps
//    - !insideConstrainedOut branch (UnconstrainedChunk):
//      UnconstrainedChunk costs stepsUsed steps and generatedOut satisfies
//      |generatedOut| <= |generated| + stepsUsed. We add stepsUsed to steps.
//      So |generated| <= |generatedPrefix| + steps is preserved.
//      EnterObservedConstrainedSpan costs 0 and doesn't change generated, so no effect.
//    - insideConstrainedOut branch (CloseSpanWithinBudget):
//      We pass closeBudget = maxSteps - steps as the budget. CloseSpanWithinBudget
//      guarantees |generatedOut| <= |generated| + closeBudget. We then set
//      steps := steps + closeBudget (= maxSteps). So |generated| <=
//      |generatedPrefix| + (steps + closeBudget) = |generatedPrefix| + maxSteps.
//      The invariant |generated| <= |generatedPrefix| + steps is maintained because
//      steps = maxSteps after this.
//
// Progress condition (last ensures):
//   If maxSteps > 0, the loop takes at least one step (either UnconstrainedChunk
//   uses at least 1 step, or CloseSpanWithinBudget uses at least 1 step).
//   So cost > 0 when maxSteps > 0, unless we start inside a constrained span
//   and remaining == 0 (but then steps == maxSteps >= 1 would require maxSteps >= 1
//   and the loop would enter the insideConstrainedOut branch and set steps := maxSteps,
//   spending at least 1 step).
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve math problems step by step. For each intermediate calculation and the FINAL ANSWER, use exactly << expr >> format. Inside << >> write ONLY a bare symbolic expression using variable names (without braces), numbers, +, -, *, /, //, %, (, ) and spaces. NO curly braces, NO **, NO text, NO words inside << >>. Correct: <<n * price + extra>>. Wrong: <<{n} * {price}>>. Always close << with >>.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;

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
    // Generate freely until we hit "<<" or EOS
    // UnconstrainedChunk stops naturally when it observes "<<" in the output
    var remaining: nat := maxSteps - steps;
    // Reserve at least 10 steps for constrained span closure
    var chunkBudget: nat := if remaining > 10 then remaining - 10 else remaining;
    if chunkBudget == 0 {
      chunkBudget := remaining;
    }
    var chunkGenerated, stoppedOnOpenSpan, stoppedOnEos, stepsUsed :=
      helpers.UnconstrainedChunk(lm, prompt, generated, chunkBudget, "<<", eosToken);
    steps := steps + stepsUsed;
    generated := chunkGenerated;

    if stoppedOnOpenSpan {
      // "<<" was already appended to generated by UnconstrainedChunk
      // Enter constrained mode without appending another "<<"
      generated, insideConstrainedOut, currentConstrainedOut :=
        helpers.EnterObservedConstrainedSpan(lm, generated);
    } else {
      // Hit EOS or chunk budget exhausted without seeing "<<"
      if stoppedOnEos {
        break;
      }
      // If we used the reserved budget path (chunkBudget < remaining) and didn't open a span,
      // we might still have steps left - but we're outside a span and should continue or stop.
      // The loop condition will handle continuing.
    }
  } else {
    // Inside a constrained span: use CloseSpanWithinBudget to generate valid content and close
    var remaining: nat := maxSteps - steps;
    if remaining == 0 {
      break;
    }
    // Use all remaining budget to close the span with parser-valid content
    var closeBudget: nat := remaining;
    var cg, ci, cc := helpers.CloseSpanWithinBudget(
      lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
    );
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + closeBudget;
  }
}

cost := steps;

