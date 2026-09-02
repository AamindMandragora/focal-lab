// CSD_RATIONALE_BEGIN
// Key insight from evaluation feedback:
// - Only 3/49 examples actually used the constrained branch
// - The model emits "<<" naturally but the reactive trigger rarely fires
// - Fix: FORCE span entry by using UnconstrainedChunk with "<<" as stop token,
//   then call EnterObservedConstrainedSpan when "<<" is detected
//
// Strategy:
// 1. Use UnconstrainedChunk to generate free text until "<<" appears
// 2. When "<<" is observed in the output, call EnterObservedConstrainedSpan
//    (which marks us as inside constrained mode without appending another "<<")
// 3. Inside constrained spans, use SafeRepetitionPenaltyStep for generation
//    with CloseSpanIfComplete to exit cleanly
// 4. Keep spans short to avoid syntax failures
//
// The critical change: UnconstrainedChunk returns stoppedOnOpenSpan=true when
// "<<" appears, and generatedOut already ends with "<<". We then call
// EnterObservedConstrainedSpan to flip into constrained mode correctly.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   Outside-span via UnconstrainedChunk: if stoppedOnOpenSpan is true,
//     generatedOut ends with "<<" already. We call EnterObservedConstrainedSpan
//     which sets insideConstrainedOut := true and currentConstrainedOut := [],
//     which is valid by parser.IsValidPrefix([]).
//     If stoppedOnEos or normal stop, insideConstrainedOut stays false (vacuous).
//   CloseSpanIfComplete (closed=true): insideConstrainedOut becomes false,
//     implication vacuous. (closed=false): state unchanged.
//   Constrained step branch: SafeRepetitionPenaltyStep returns EOS or parser-
//     valid extension. AppendConstrainedToken preserves IsValidPrefix by contract.
//
// progress:
//   UnconstrainedChunk branch: steps += stepsUsed; generated grows by at most
//     stepsUsed tokens (stepsUsed <= maxChunkSize <= maxSteps - steps).
//   CloseSpanIfComplete branch: steps += 1; generated grows by at most 1.
//   Constrained step branch: steps += 1; generated grows by at most 1.
//   All branches preserve |generated| <= |generatedPrefix| + steps.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve the math word problem step by step. Use << >> to wrap each symbolic expression and the final answer. Write only one expression per << >> span. Keep each << >> span short and exact.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;
var MAX_SPAN_TOKENS: nat := 40;
var spanTokenCount: nat := 0;

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
    // Use UnconstrainedChunk to generate until "<<" or EOS
    var maxChunk: nat := if maxSteps - steps >= 20 then 20 else maxSteps - steps;
    if maxChunk == 0 {
      break;
    }
    var genOut, stoppedOnOpenSpan, stoppedOnEos, stepsUsed :=
      helpers.UnconstrainedChunk(lm, prompt, generated, maxChunk, "<<", eosToken);
    // stepsUsed <= maxChunk <= maxSteps - steps
    steps := steps + stepsUsed;
    generated := genOut;
    if stoppedOnEos {
      break;
    } else if stoppedOnOpenSpan {
      // "<<" is already in generated; enter constrained mode
      var eg, ei, ec := helpers.EnterObservedConstrainedSpan(lm, generated);
      generated := eg;
      insideConstrainedOut := ei;
      currentConstrainedOut := ec;
      spanTokenCount := 0;
    }
    // else: chunk completed without "<<" or EOS; loop again
  } else if spanTokenCount >= MAX_SPAN_TOKENS {
    // Span too long - roll back to last complete and close
    var rg, rc := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
    generated := rg;
    currentConstrainedOut := rc;
    if parser.IsCompletePrefix(currentConstrainedOut) {
      var cg2, ci2, cc2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := cg2;
      insideConstrainedOut := ci2;
      currentConstrainedOut := cc2;
      steps := steps + 1;
      spanTokenCount := 0;
    } else {
      // Empty prefix after rollback - just exit constrained mode by breaking
      // We need to charge a step and exit
      insideConstrainedOut := false;
      currentConstrainedOut := [];
      steps := steps + 1;
      spanTokenCount := 0;
    }
  } else {
    // Try to close if complete first
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    steps := steps + 1;
    if closed {
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      spanTokenCount := 0;
    } else {
      // Generate next constrained token
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.SafeRepetitionPenaltyStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, generated, 2.0, eosToken
      );
      if next == eosToken {
        break;
      } else {
        var ag, ai, ac := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := ag;
        insideConstrainedOut := ai;
        currentConstrainedOut := ac;
        spanTokenCount := spanTokenCount + 1;
      }
    }
  }
}

cost := steps;
