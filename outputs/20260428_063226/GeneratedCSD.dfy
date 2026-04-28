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
    var answer_tokens := 0;
    var close_score := 0;
    var reason_limit := 2;
    while ((stepsLeft > 0) && (phase < 4))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if ((phase == 0) && (reason_steps < reason_limit) && (stepsLeft > 0)) {
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        reason_steps := (reason_steps + 1);
        if reason_steps >= reason_limit {
          phase := 1;
        }
      } else {
        if ((phase == 0) && (stepsLeft > 0)) {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          phase := 1;
        } else {
          if ((phase == 1) && (stepsLeft > 0)) {
            generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
            phase := 2;
          } else {
            if ((phase == 2) && (helpers.CanConstrain(generated))) {
              generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
              answer_tokens := (answer_tokens + 1);
              if answer_tokens >= 2 {
                close_score := (close_score + 1);
              }
            } else {
              if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))) && (helpers.CanExtendConstrained(generated)) && (answer_tokens < 3)) {
                generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                answer_tokens := (answer_tokens + 1);
                close_score := (close_score + 1);
              } else {
                if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))) && (helpers.CanExtendConstrained(generated)) && (parser.ValidContinuationCount(helpers.LongestValidSuffix(generated)) > 1) && ((helpers.MinStepsToComplete(generated) + 1) < stepsLeft) && (close_score < 2)) {
                  generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                  answer_tokens := (answer_tokens + 1);
                  close_score := (close_score + 1);
                } else {
                  if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))) && (answer_tokens > 0) && (((close_score > 0) || (stepsLeft <= 2) || (parser.ValidContinuationCount(helpers.LongestValidSuffix(generated)) <= 1)))) {
                    generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                    phase := 3;
                  } else {
                    if ((phase == 3) && (stepsLeft > 0)) {
                      generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                      phase := 4;
                    } else {
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}