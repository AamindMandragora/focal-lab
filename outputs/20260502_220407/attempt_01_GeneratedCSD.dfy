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
    // Use a durable late-open final-span policy for GSM symbolic reasoning.
    // Stay in delimiter-masked free-form reasoning for a substantial setup period,
    // then enter a short wrap-up phase, then persistently nudge for a natural left
    // delimiter. Once the left delimiter appears, keep explicit inside-span state
    // and use constrained-or-right-delimiter steps until the span closes. After the
    // first closed span, stop, making that closed verified span the final answer.
    // This respects GSM natural-delimiter rules, avoids explicit delimiter forcing,
    // and ensures every non-break branch consumes a helper step.
    // CSD_RATIONALE_END
    // CSD_PROOF_SKETCH_BEGIN
    // The loop preserves the required invariants because each helper step returns a
    // new generated sequence and reduced stepsLeft, while the helper object fields
    // remain unchanged. We never manually decrement stepsLeft. State transitions
    // into wrap/open/span modes occur only in branches that also take a helper
    // step. The inside-span branch uses the positive guard
    // helpers.IsComplete(generated) or helpers.CanConstrain(generated) before
    // calling AppendConstrainedOrRightDelimiterStep, so complete expressions may
    // close with >> and incomplete valid prefixes may continue. closed_spans is a
    // real counter updated only when a right delimiter is actually observed. After
    // the first closed span, the strategy breaks, yielding a single late final
    // verified span.
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
      } else if phase == "open" {
        generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
        nudge_steps := nudge_steps + 1;
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          phase := "span";
        }
      } else if phase == "wrap" {
        if helpers.HasBudget(stepsLeft, 20) {
          generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
          nudge_steps := nudge_steps + 1;
          if helpers.EndsWithLeftDelimiter(generated) {
            inside_span := true;
            phase := "span";
          } else {
            phase := "open";
          }
        } else {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          wrap_steps := wrap_steps + 1;
        }
      } else {
        if reason_steps >= 40 {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          wrap_steps := wrap_steps + 1;
          phase := "wrap";
        } else {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          reason_steps := reason_steps + 1;
        }
      }
    }
    remainingSteps := stepsLeft;
  }
}
