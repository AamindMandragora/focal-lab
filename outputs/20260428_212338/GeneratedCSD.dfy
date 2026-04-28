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
    var answer_ready := 0;
    var closed_spans := 0;
    var inside_steps := 0;
    var close_pressure := 0;
    var next_token := eosToken;
    var new_steps := stepsLeft;
    while ((stepsLeft > 0) && (phase < 3) && (closed_spans < 2))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if ((phase == 0) && (answer_ready == 0) && (helpers.HasBudget(stepsLeft, 12))) {
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        reason_steps := (reason_steps + 1);
        if reason_steps >= 8 {
          answer_ready := 1;
        }
      } else {
        if phase == 0 {
          next_token, new_steps := helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
          generated := (generated + [next_token]);
          stepsLeft := new_steps;
          if ((next_token == LeftDelimiter) || (next_token == SpacedLeftDelimiter)) {
            phase := 1;
            inside_steps := 0;
            close_pressure := 0;
          } else {
            reason_steps := (reason_steps + 1);
            if reason_steps >= 12 {
              answer_ready := 1;
            }
          }
        } else {
          if ((phase == 1) && (helpers.CanConstrain(generated))) {
            generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
            inside_steps := (inside_steps + 1);
            if inside_steps >= 3 {
              close_pressure := 1;
            }
          } else {
            if ((phase == 1) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
              if ((close_pressure > 0) || (!helpers.HasBudget(stepsLeft, 3)) || (parser.ValidContinuationCount(helpers.LongestValidSuffix(generated)) <= 1)) {
                next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                generated := (generated + [next_token]);
                stepsLeft := new_steps;
                if ((next_token == RightDelimiter) || (next_token == SpacedRightDelimiter)) {
                  closed_spans := (closed_spans + 1);
                  phase := 2;
                } else {
                  inside_steps := (inside_steps + 1);
                  close_pressure := 1;
                }
              } else {
                next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                generated := (generated + [next_token]);
                stepsLeft := new_steps;
                if ((next_token == RightDelimiter) || (next_token == SpacedRightDelimiter)) {
                  closed_spans := (closed_spans + 1);
                  phase := 2;
                } else {
                  inside_steps := (inside_steps + 1);
                  if inside_steps >= 4 {
                    close_pressure := 1;
                  }
                }
              }
            } else {
              if phase == 2 {
                break;
              } else {
                break;
              }
            }
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}