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
    var free_steps := 0;
    var closed_spans := 0;
    var post_span_free_steps := 0;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if phase == 2 {
        if helpers.EndsWithRightDelimiter(generated) {
          closed_spans := (closed_spans + 1);
          if closed_spans >= 2 {
            break;
          }
          phase := 0;
          post_span_free_steps := 0;
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          free_steps := (free_steps + 1);
          post_span_free_steps := (post_span_free_steps + 1);
        } else {
          if !helpers.CanConstrain(generated) {
            break;
          } else {
            if helpers.IsComplete(generated) {
              generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
            } else {
              generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
            }
          }
        }
      } else {
        if phase == 1 {
          if helpers.EndsWithLeftDelimiter(generated) {
            phase := 2;
            if !helpers.CanConstrain(generated) {
              break;
            } else {
              if helpers.IsComplete(generated) {
                generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
              } else {
                generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
              }
            }
          } else {
            generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
            if helpers.EndsWithLeftDelimiter(generated) {
              phase := 2;
            }
          }
        } else {
          if closed_spans == 0 {
            if free_steps >= 24 {
              phase := 1;
              generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
              if helpers.EndsWithLeftDelimiter(generated) {
                phase := 2;
              }
            } else {
              if ((free_steps >= 16) && (!helpers.HasBudget(stepsLeft, 10))) {
                phase := 1;
                generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                if helpers.EndsWithLeftDelimiter(generated) {
                  phase := 2;
                }
              } else {
                if ((free_steps >= 12) && (!helpers.HasBudget(stepsLeft, 6))) {
                  phase := 1;
                  generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                  if helpers.EndsWithLeftDelimiter(generated) {
                    phase := 2;
                  }
                } else {
                  if ((free_steps >= 8) && (!helpers.HasBudget(stepsLeft, 4))) {
                    phase := 1;
                    generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                    if helpers.EndsWithLeftDelimiter(generated) {
                      phase := 2;
                    }
                  } else {
                    generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                    free_steps := (free_steps + 1);
                  }
                }
              }
            }
          } else {
            if post_span_free_steps >= 16 {
              phase := 1;
              generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
              if helpers.EndsWithLeftDelimiter(generated) {
                phase := 2;
              }
            } else {
              if ((post_span_free_steps >= 8) && (!helpers.HasBudget(stepsLeft, 8))) {
                phase := 1;
                generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                if helpers.EndsWithLeftDelimiter(generated) {
                  phase := 2;
                }
              } else {
                if ((post_span_free_steps >= 4) && (!helpers.HasBudget(stepsLeft, 5))) {
                  phase := 1;
                  generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                  if helpers.EndsWithLeftDelimiter(generated) {
                    phase := 2;
                  }
                } else {
                  generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                  free_steps := (free_steps + 1);
                  post_span_free_steps := (post_span_free_steps + 1);
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