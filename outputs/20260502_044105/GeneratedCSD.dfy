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
    var reasoning_steps := 0;
    var closed_spans := 0;
    var moderate_reasoning_threshold := 8;
    var late_reasoning_threshold := 14;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      var stepsLeftBeforeIteration := stepsLeft;
      if closed_spans > 0 {
        break;
      } else {
        if phase == "reason" {
          if helpers.EndsWithRightDelimiter(generated) {
            closed_spans := (closed_spans + 1);
            break;
          } else {
            if helpers.EndsWithLeftDelimiter(generated) {
              phase := "inside_span";
              if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                if helpers.EndsWithRightDelimiter(generated) {
                  closed_spans := (closed_spans + 1);
                  break;
                }
              } else {
                break;
              }
            } else {
              var answer_ready := false;
              if reasoning_steps >= late_reasoning_threshold {
                answer_ready := true;
              } else {
                if reasoning_steps >= moderate_reasoning_threshold {
                  if helpers.IsComplete(generated) {
                    answer_ready := true;
                  } else {
                    if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                      answer_ready := true;
                    } else {
                      if helpers.ValidContinuationCount(generated) <= 1 {
                        answer_ready := true;
                      } else {
                        if helpers.ParserDistanceToComplete(generated) <= 1 {
                          answer_ready := true;
                        } else {
                          if helpers.MinStepsToComplete(generated) <= 1 {
                            answer_ready := true;
                          }
                        }
                      }
                    }
                  }
                }
              }
              if answer_ready {
                phase := "nudge_left";
                generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                if helpers.EndsWithRightDelimiter(generated) {
                  closed_spans := (closed_spans + 1);
                  break;
                } else {
                  if helpers.EndsWithLeftDelimiter(generated) {
                    phase := "inside_span";
                  } else {
                    phase := "nudge_left";
                  }
                }
              } else {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                reasoning_steps := (reasoning_steps + 1);
                if helpers.EndsWithRightDelimiter(generated) {
                  closed_spans := (closed_spans + 1);
                  break;
                } else {
                  if helpers.EndsWithLeftDelimiter(generated) {
                    phase := "inside_span";
                  } else {
                    phase := "reason";
                  }
                }
              }
            }
          }
        } else {
          if phase == "nudge_left" {
            if helpers.EndsWithRightDelimiter(generated) {
              closed_spans := (closed_spans + 1);
              break;
            } else {
              if helpers.EndsWithLeftDelimiter(generated) {
                phase := "inside_span";
                if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                  generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                  if helpers.EndsWithRightDelimiter(generated) {
                    closed_spans := (closed_spans + 1);
                    break;
                  }
                } else {
                  break;
                }
              } else {
                generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                if helpers.EndsWithRightDelimiter(generated) {
                  closed_spans := (closed_spans + 1);
                  break;
                } else {
                  if helpers.EndsWithLeftDelimiter(generated) {
                    phase := "inside_span";
                  } else {
                    phase := "nudge_left";
                  }
                }
              }
            }
          } else {
            if phase == "inside_span" {
              if helpers.EndsWithRightDelimiter(generated) {
                closed_spans := (closed_spans + 1);
                break;
              } else {
                if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                  generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                  if helpers.EndsWithRightDelimiter(generated) {
                    closed_spans := (closed_spans + 1);
                    break;
                  } else {
                    phase := "inside_span";
                  }
                } else {
                  break;
                }
              }
            } else {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              if helpers.EndsWithRightDelimiter(generated) {
                closed_spans := (closed_spans + 1);
                break;
              } else {
                if helpers.EndsWithLeftDelimiter(generated) {
                  phase := "inside_span";
                } else {
                  phase := "reason";
                }
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