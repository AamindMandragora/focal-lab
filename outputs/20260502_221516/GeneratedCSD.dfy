include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, remainingSteps: nat)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires !parser.IsCompletePrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires maxSteps >= 2
    requires LeftDelimiter in lm.Tokens
    requires RightDelimiter in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures remainingSteps >= 0 && remainingSteps <= maxSteps
  {
    var helpers := new CSDHelpers(lm, parser);
    lm.ValidTokensIdsLogitsAlways();
    generated := [];
    var stepsLeft := maxSteps;
    // CSD_RATIONALE_BEGIN
    // Use a durable late-open GSM policy: keep delimiters masked during ordinary reasoning,
    // then enter a short wrap-up phase, then persistently nudge for a natural `<<` opening.
    // Once the left delimiter appears, remain in explicit span mode until `>>` closes a
    // grammar-valid arithmetic expression or equation. Prefer exactly one late final span.
    // Closed spans are tracked with a real counter, and after the first closed span we stop.
    // CSD_RATIONALE_END
    // CSD_PROOF_SKETCH_BEGIN
    // The loop preserves the required invariants because every non-break branch calls one
    // helper that returns an updated `stepsLeft`, and no branch manually decrements it.
    // `inside_span` is the durable indicator for constrained mode; `EndsWithLeftDelimiter`
    // is treated only as the opening event that sets `inside_span := true` after the same
    // helper step that emitted it. Inside a span, we use the positive guard
    // `helpers.IsComplete(generated) || helpers.CanConstrain(generated)` before calling
    // `AppendConstrainedOrRightDelimiterStep`, so complete expressions may close with `>>`.
    // After the first right delimiter, `closed_spans` is incremented and the strategy breaks,
    // yielding a single late final verified span.
    // CSD_PROOF_SKETCH_END
    var phase := "reason";
    var inside_span := false;
    var closed_spans := 0;
    var reason_steps := 0;
    var wrap_steps := 0;
    var nudge_steps := 0;

    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if inside_span {
        if helpers.IsComplete(generated) || helpers.CanConstrain(generated) {
          generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
          if helpers.EndsWithRightDelimiter(generated) {
            inside_span := false;
            closed_spans := closed_spans + 1;
            phase := "done";
          }
        } else {
          break;
        }
      } else if phase == "done" {
        break;
      } else if phase == "reason" {
        if reason_steps < 40 && helpers.HasBudget(stepsLeft, 20) {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          reason_steps := reason_steps + 1;
        } else {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          wrap_steps := 1;
          phase := "wrap";
        }
      } else if phase == "wrap" {
        if wrap_steps < 3 && helpers.HasBudget(stepsLeft, 12) {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          wrap_steps := wrap_steps + 1;
        } else {
          generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
          nudge_steps := nudge_steps + 1;
          phase := "open";
          if helpers.EndsWithLeftDelimiter(generated) {
            inside_span := true;
          }
        }
      } else {
        generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
        nudge_steps := nudge_steps + 1;
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
        }
      }

      if closed_spans > 0 {
        break;
      }
    }
    remainingSteps := stepsLeft;
  }
}
