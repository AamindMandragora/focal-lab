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
    // Use a durable late-open GSM natural-delimiter policy.
    // Stay in ordinary delimiter-masked free-form reasoning for a substantial setup period.
    // After enough setup, enter a short wrap-up phase, then persistently nudge toward a natural
    // `<<` opening. Once the left delimiter appears, keep explicit inside-span state and use
    // constrained-or-right-delimiter steps until the span naturally closes with `>>`.
    // Prefer a single late final symbolic arithmetic span, then stop after the first closed span.
    // This matches the observed strong GSM policy of delaying the final verified expression until
    // the answer is ready, while still allowing the parser to ensure the final span is grammar-valid.
    // CSD_RATIONALE_END
    // CSD_PROOF_SKETCH_BEGIN
    // The loop preserves the required invariants because every non-break branch consumes exactly one
    // helper step, and helper-step methods return an updated `stepsLeft` that respects the decoding
    // budget. No branch manually decrements `stepsLeft`.
    // `inside_span` is the durable indicator that the strategy is currently decoding a constrained
    // GSM span; `helpers.EndsWithLeftDelimiter(generated)` is treated only as the opening event.
    // Inside a span, the positive guard `helpers.IsComplete(generated) || helpers.CanConstrain(generated)`
    // is checked before calling `AppendConstrainedOrRightDelimiterStep`, so complete expressions may
    // close naturally with `>>` and incomplete valid prefixes may continue.
    // After the first closed span, the strategy breaks, yielding a single late final verified span.
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
        if reason_steps < 40 {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          reason_steps := reason_steps + 1;
        } else {
          phase := "wrap";
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          wrap_steps := 1;
        }
      } else if phase == "wrap" {
        if wrap_steps < 3 {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          wrap_steps := wrap_steps + 1;
        } else {
          phase := "nudge";
          generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
          nudge_steps := 1;
          if helpers.EndsWithLeftDelimiter(generated) {
            inside_span := true;
            phase := "span";
          }
        }
      } else if phase == "nudge" {
        generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
        nudge_steps := nudge_steps + 1;
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          phase := "span";
        }
      } else {
        generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          phase := "span";
        }
      }
    }
    remainingSteps := stepsLeft;
  }
}
