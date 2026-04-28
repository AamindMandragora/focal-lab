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
    var phase := 0;
    var reason_steps := 0;
    var max_reason_steps := 24;
    var min_reason_steps := 8;
    var answer_steps := 0;
    var min_answer_steps := 3;
    var max_answer_steps := 12;
    var close_slack := 2;
    while ((stepsLeft > 0) && (phase < 3))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if phase == 0 {
        var need_for_answer := 1;
        if helpers.HasBudget(stepsLeft, (close_slack + 2)) {
          need_for_answer := 0;
        }
        if ((reason_steps < min_reason_steps) && (need_for_answer == 0)) {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          reason_steps := (reason_steps + 1);
        } else {
          if ((reason_steps < max_reason_steps) && (need_for_answer == 0) && (stepsLeft > ((helpers.MinStepsToComplete(generated) + close_slack) + 1))) {
            generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
            reason_steps := (reason_steps + 1);
          } else {
            generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
            phase := 1;
          }
        }
      } else {
        if ((phase == 1) && (helpers.CanConstrain(generated))) {
          if stepsLeft <= (close_slack + 2) {
            generated, stepsLeft := helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft);
          } else {
            generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
          }
          answer_steps := (answer_steps + 1);
        } else {
          if ((phase == 1) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
            var suffix := helpers.LongestValidSuffix(generated);
            var continuation_count := parser.ValidContinuationCount(suffix);
            var should_close := 0;
            if answer_steps >= min_answer_steps {
              should_close := 1;
            }
            if continuation_count == 0 {
              should_close := 1;
            }
            if answer_steps >= max_answer_steps {
              should_close := 1;
            }
            if stepsLeft <= 1 {
              should_close := 1;
            }
            if should_close == 1 {
              generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
              phase := 2;
            } else {
              if helpers.CanExtendConstrained(generated) {
                generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                answer_steps := (answer_steps + 1);
              } else {
                generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                phase := 2;
              }
            }
          } else {
            if phase == 2 {
              if stepsLeft > 0 {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              } else {
                break;
              }
            } else {
              break;
            }
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}