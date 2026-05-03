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
    // This strategy uses GSM natural-delimiter control with a durable late-open final-span policy.
    // It keeps most early decoding unconstrained so the model can produce ordinary reasoning text.
    // After substantial setup, an explicit wrap-up phase, or real budget pressure, it persistently
    // nudges toward a natural `<<` opening. Once the left delimiter appears, it enters durable
    // inside-span mode and uses constrained-or-right-delimiter decoding until the span closes.
    // The policy tracks a real closed-span counter and prefers stopping after the first closed final
    // span, matching the strong GSM preference for one delayed final arithmetic expression or equation.
    // CSD_RATIONALE_END
    // CSD_PROOF_SKETCH_BEGIN
    // The loop preserves the required invariants because every non-break path consumes exactly one
    // helper step, and helper step methods return a new `stepsLeft` that is no larger than before.
    // No manual decrement is used. The inside-span branch checks the positive guard
    // `helpers.IsComplete(generated) or helpers.CanConstrain(generated)` before calling the natural
    // constrained step, so complete suffixes can emit `>>` and incomplete valid suffixes can continue.
    // State transitions into answer-opening or span mode occur only in branches that also consume a
    // helper step. After a right delimiter is observed, `closed_spans` is incremented and the phase
    // becomes `done`, after which the loop breaks. Thus the strategy is verifier-friendly and follows
    // the GSM natural-delimiter policy.
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
      if phase == "done" {
        break;
      } else if inside_span {
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
      } else if phase == "nudge" {
        generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
        nudge_steps := nudge_steps + 1;
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          phase := "span";
        }
      } else if phase == "wrap" {
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        wrap_steps := wrap_steps + 1;
        reason_steps := reason_steps + 1;
        if wrap_steps >= 5 || !helpers.HasBudget(stepsLeft, 14) {
          phase := "nudge";
        }
      } else {
        if closed_spans > 0 {
          phase := "done";
          break;
        } else if reason_steps >= 40 {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          reason_steps := reason_steps + 1;
          wrap_steps := 1;
          phase := "wrap";
        } else if !helpers.HasBudget(stepsLeft, 18) {
          generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
          nudge_steps := nudge_steps + 1;
          phase := "nudge";
          if helpers.EndsWithLeftDelimiter(generated) {
            inside_span := true;
            phase := "span";
          }
        } else {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          reason_steps := reason_steps + 1;
        }
      }
    }
    remainingSteps := stepsLeft;
  }
}
