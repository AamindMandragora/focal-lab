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
    var inside_span := false;
    var closed_spans := 0;
    var reasoning_steps := 0;
    var nudge_steps := 0;
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
        if phase == "done" {
          break;
        } else {
          if ((inside_span) || (phase == "span")) {
            if helpers.EndsWithRightDelimiter(generated) {
              inside_span := false;
              closed_spans := (closed_spans + 1);
              phase := "done";
              break;
            } else {
              if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                if helpers.EndsWithRightDelimiter(generated) {
                  inside_span := false;
                  closed_spans := (closed_spans + 1);
                  phase := "done";
                  break;
                } else {
                  inside_span := true;
                  phase := "span";
                }
              } else {
                break;
              }
            }
          } else {
            if helpers.EndsWithLeftDelimiter(generated) {
              inside_span := true;
              phase := "span";
              if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                if helpers.EndsWithRightDelimiter(generated) {
                  inside_span := false;
                  closed_spans := (closed_spans + 1);
                  phase := "done";
                  break;
                } else {
                  inside_span := true;
                  phase := "span";
                }
              } else {
                break;
              }
            } else {
              if phase == "reason" {
                var answer_ready := false;
                if reasoning_steps >= 24 {
                  answer_ready := true;
                } else {
                  if reasoning_steps >= 16 {
                    if stepsLeft <= 20 {
                      answer_ready := true;
                    } else {
                      if helpers.ValidContinuationCount(generated) <= 2 {
                        answer_ready := true;
                      } else {
                        if helpers.ParserDistanceToComplete(generated) <= 4 {
                          answer_ready := true;
                        } else {
                          if helpers.MinStepsToComplete(generated) <= 4 {
                            answer_ready := true;
                          }
                        }
                      }
                    }
                  }
                }
                if answer_ready {
                  phase := "nudge";
                  generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                  nudge_steps := (nudge_steps + 1);
                  if helpers.EndsWithLeftDelimiter(generated) {
                    inside_span := true;
                    phase := "span";
                  }
                } else {
                  generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                  reasoning_steps := (reasoning_steps + 1);
                }
              } else {
                if phase == "nudge" {
                  if helpers.EndsWithLeftDelimiter(generated) {
                    inside_span := true;
                    phase := "span";
                    if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                      generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                      if helpers.EndsWithRightDelimiter(generated) {
                        inside_span := false;
                        closed_spans := (closed_spans + 1);
                        phase := "done";
                        break;
                      } else {
                        inside_span := true;
                        phase := "span";
                      }
                    } else {
                      break;
                    }
                  } else {
                    if stepsLeft <= 6 {
                      break;
                    } else {
                      generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                      nudge_steps := (nudge_steps + 1);
                      if helpers.EndsWithLeftDelimiter(generated) {
                        inside_span := true;
                        phase := "span";
                      }
                    }
                  }
                } else {
                  break;
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