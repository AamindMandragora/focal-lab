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
    var answer_steps := 0;
    var min_reason_steps := 40;
    var target_reason_steps := 56;
    var min_answer_steps := 12;
    var max_answer_steps := 24;
    var pressure := 0;
    var close_ready := 0;
    while ((stepsLeft > 0) && (phase < 3))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if phase == 0 {
        var need_for_answer := (min_answer_steps + 2);
        if reason_steps < min_reason_steps {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          reason_steps := (reason_steps + 1);
        } else {
          if !helpers.HasBudget(stepsLeft, need_for_answer) {
            generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
            phase := 1;
          } else {
            if ((reason_steps < target_reason_steps) && (pressure < 2)) {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              reason_steps := (reason_steps + 1);
              if !helpers.HasBudget(stepsLeft, (min_answer_steps + 4)) {
                pressure := (pressure + 1);
              }
            } else {
              generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
              phase := 1;
            }
          }
        }
      } else {
        if phase == 1 {
          if helpers.CanConstrain(generated) {
            generated, stepsLeft := helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft);
            answer_steps := (answer_steps + 1);
            phase := 2;
          } else {
            break;
          }
        } else {
          if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
            if answer_steps >= min_answer_steps {
              close_ready := 1;
            }
            if ((close_ready > 0) && (((answer_steps >= max_answer_steps) || (!helpers.CanExtendConstrained(generated)) || (!helpers.HasBudget(stepsLeft, 1))))) {
              generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
              phase := 3;
            } else {
              if helpers.CanExtendConstrained(generated) {
                generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                answer_steps := (answer_steps + 1);
              } else {
                generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                phase := 3;
              }
            }
          } else {
            if ((phase == 2) && (helpers.CanConstrain(generated)) && (!parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
              if answer_steps < 4 {
                generated, stepsLeft := helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft);
              } else {
                if (helpers.MinStepsToComplete(generated) + 1) >= stepsLeft {
                  generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
                } else {
                  generated, stepsLeft := helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft);
                }
              }
              answer_steps := (answer_steps + 1);
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