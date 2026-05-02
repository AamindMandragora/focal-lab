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
    // Use a durable late-open policy for GSM symbolic decoding.
    // Stay in delimiter-masked free-form reasoning for a substantial prefix, then
    // enter a short wrap-up phase, and only then persistently nudge for a single
    // natural final << >> span. Inside the span, use constrained-or-close steps so
    // the model can produce a compact grammar-valid arithmetic expression or
    // equation and naturally emit >> when complete. After the first closed span,
    // stop, making that final span the graded answer.
    // CSD_RATIONALE_END
    // CSD_PROOF_SKETCH_BEGIN
    // The loop preserves the required helper/object invariants and decreases only
    // through helper calls that return an updated stepsLeft. Every non-break branch
    // consumes exactly one helper step. State tracks whether we are in free-form
    // reasoning, wrap-up, opening pressure, or inside the verified span. Opening is
    // detected by EndsWithLeftDelimiter(generated) as an event, after which
    // inside_span becomes true. While inside_span, we only call
    // AppendConstrainedOrRightDelimiterStep when IsComplete(generated) or
    // CanConstrain(generated), ensuring complete expressions may close and
    // incomplete ones may continue. When EndsWithRightDelimiter(generated) holds,
    // we count the closed span and break, yielding one late final verified span.
    // CSD_PROOF_SKETCH_END
    var phase := "reason";
    var inside_span := false;
    var closed_spans := 0;
    var reason_steps := 0;
    var wrap_steps := 0;
    var nudge_steps := 0;
    var saw_answer_cue := false;

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
            break;
          }
        } else {
          break;
        }
      } else if phase == "open" {
        generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
        nudge_steps := nudge_steps + 1;
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          phase := "span";
        }
      } else if phase == "wrap" {
        if saw_answer_cue || reason_steps >= 40 || !helpers.HasBudget(stepsLeft, 20) {
          generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
          nudge_steps := nudge_steps + 1;
          phase := "open";
          if helpers.EndsWithLeftDelimiter(generated) {
            inside_span := true;
            phase := "span";
          }
        } else {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          wrap_steps := wrap_steps + 1;
          if wrap_steps >= 14 {
            saw_answer_cue := true;
          }
        }
      } else {
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        reason_steps := reason_steps + 1;
        if reason_steps >= 44 || !helpers.HasBudget(stepsLeft, 28) {
          phase := "wrap";
          if reason_steps >= 52 {
            saw_answer_cue := true;
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }
}
