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
    var phase := "reason";
    var in_span := false;
    var closed_spans := 0;
    var reasoning_steps := 0;
    var nudge_steps := 0;
    var answer_mode := false;
    while ((stepsLeft > 0) && (closed_spans == 0))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      var stepsLeftBeforeIteration := stepsLeft;
      if in_span {
        if helpers.EndsWithRightDelimiter(generated) {
          in_span := false;
          closed_spans := (closed_spans + 1);
          phase := "done";
          break;
        } else {
          if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
            phase := "span";
            generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
          } else {
            if helpers.IsDead(generated) {
              break;
            } else {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
            }
          }
        }
      } else {
        if helpers.EndsWithLeftDelimiter(generated) {
          in_span := true;
          phase := "span";
          generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
        } else {
          var distance := helpers.ParserDistanceToComplete(generated);
          var min_steps := helpers.MinStepsToComplete(generated);
          var continuation_count := helpers.ValidContinuationCount(generated);
          var enough_reasoning := reasoning_steps >= 8;
          var parser_ready := ((distance <= 4) || (min_steps <= 4));
          var budget_ready := stepsLeft <= 18;
          if ((!answer_mode) && (((enough_reasoning) || (parser_ready) || (budget_ready)))) {
            answer_mode := true;
            phase := "seek_open";
            generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
            nudge_steps := (nudge_steps + 1);
          } else {
            if ((answer_mode) && (!helpers.EndsWithLeftDelimiter(generated))) {
              if ((nudge_steps < 6) || (continuation_count > 0)) {
                phase := "seek_open";
                generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                nudge_steps := (nudge_steps + 1);
              } else {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                reasoning_steps := (reasoning_steps + 1);
              }
            } else {
              if helpers.IsDead(generated) {
                break;
              } else {
                phase := "reason";
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                reasoning_steps := (reasoning_steps + 1);
              }
            }
          }
        }
      }
      if stepsLeft >= stepsLeftBeforeIteration {
        break;
      }
    }
    remainingSteps := stepsLeft;
  }

}