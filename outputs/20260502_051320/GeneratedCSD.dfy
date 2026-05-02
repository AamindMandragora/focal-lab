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
    var closed_spans := 0;
    while ((stepsLeft > 0) && (closed_spans == 0))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      var stepsLeftBeforeIteration := stepsLeft;
      if phase == "reason" {
        if helpers.EndsWithRightDelimiter(generated) {
          closed_spans := (closed_spans + 1);
          if closed_spans > 0 {
            break;
          }
        } else {
          if helpers.EndsWithLeftDelimiter(generated) {
            phase := "span";
            break;
          } else {
            if helpers.IsComplete(generated) {
              phase := "open";
              break;
            } else {
              if helpers.MinStepsToComplete(generated) >= stepsLeft {
                phase := "open";
                break;
              } else {
                if (helpers.ParserDistanceToComplete(generated) + 1) >= stepsLeft {
                  phase := "open";
                  break;
                } else {
                  if helpers.ValidContinuationCount(generated) == 0 {
                    phase := "open";
                    break;
                  } else {
                    generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                    if helpers.EndsWithRightDelimiter(generated) {
                      closed_spans := (closed_spans + 1);
                    } else {
                      if helpers.EndsWithLeftDelimiter(generated) {
                        phase := "span";
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if phase == "open" {
          if helpers.EndsWithRightDelimiter(generated) {
            closed_spans := (closed_spans + 1);
            if closed_spans > 0 {
              break;
            }
          } else {
            if helpers.EndsWithLeftDelimiter(generated) {
              phase := "span";
              break;
            } else {
              generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
              if helpers.EndsWithRightDelimiter(generated) {
                closed_spans := (closed_spans + 1);
              } else {
                if helpers.EndsWithLeftDelimiter(generated) {
                  phase := "span";
                }
              }
            }
          }
        } else {
          if phase == "span" {
            if helpers.EndsWithRightDelimiter(generated) {
              closed_spans := (closed_spans + 1);
              if closed_spans > 0 {
                break;
              }
            } else {
              if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                if helpers.EndsWithRightDelimiter(generated) {
                  closed_spans := (closed_spans + 1);
                  if closed_spans > 0 {
                    break;
                  }
                }
              } else {
                break;
              }
            }
          } else {
            break;
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