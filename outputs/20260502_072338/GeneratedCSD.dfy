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
    var reason_steps := 0;
    var open_attempts := 0;
    var answer_ready := false;
    var durable_setup_steps := 44;
    var very_long_setup_steps := 60;
    var max_open_attempts := 16;
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
          if in_span {
            phase := "span";
            if helpers.EndsWithRightDelimiter(generated) {
              in_span := false;
              closed_spans := (closed_spans + 1);
              phase := "done";
              break;
            } else {
              if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                if helpers.EndsWithRightDelimiter(generated) {
                  in_span := false;
                  closed_spans := (closed_spans + 1);
                  phase := "done";
                } else {
                  in_span := true;
                  phase := "span";
                }
              } else {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                if helpers.EndsWithRightDelimiter(generated) {
                  in_span := false;
                  closed_spans := (closed_spans + 1);
                  phase := "done";
                } else {
                  in_span := true;
                  phase := "span";
                }
              }
            }
          } else {
            if phase == "reason" {
              if reason_steps >= very_long_setup_steps {
                answer_ready := true;
              } else {
                if reason_steps >= durable_setup_steps {
                  if helpers.IsComplete(generated) {
                    answer_ready := true;
                  } else {
                    if ((helpers.CanConstrain(generated)) && (helpers.ValidContinuationCount(generated) <= 2)) {
                      answer_ready := true;
                    } else {
                      if ((helpers.ParserDistanceToComplete(generated) <= 2) && (helpers.MinStepsToComplete(generated) <= 2)) {
                        answer_ready := true;
                      }
                    }
                  }
                }
              }
              if answer_ready {
                phase := "open";
                generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                open_attempts := (open_attempts + 1);
                if helpers.EndsWithLeftDelimiter(generated) {
                  in_span := true;
                  phase := "span";
                } else {
                  in_span := false;
                  phase := "open";
                }
              } else {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                reason_steps := (reason_steps + 1);
                if helpers.EndsWithLeftDelimiter(generated) {
                  in_span := true;
                  phase := "span";
                } else {
                  phase := "reason";
                }
              }
            } else {
              if phase == "open" {
                if helpers.EndsWithLeftDelimiter(generated) {
                  in_span := true;
                  phase := "span";
                  generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                  if helpers.EndsWithRightDelimiter(generated) {
                    in_span := false;
                    closed_spans := (closed_spans + 1);
                    phase := "done";
                  } else {
                    in_span := true;
                    phase := "span";
                  }
                } else {
                  if open_attempts >= max_open_attempts {
                    generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                    if helpers.EndsWithLeftDelimiter(generated) {
                      in_span := true;
                      phase := "span";
                    } else {
                      phase := "open";
                    }
                  } else {
                    generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                    open_attempts := (open_attempts + 1);
                    if helpers.EndsWithLeftDelimiter(generated) {
                      in_span := true;
                      phase := "span";
                    } else {
                      in_span := false;
                      phase := "open";
                    }
                  }
                }
              } else {
                if phase == "span" {
                  if helpers.EndsWithRightDelimiter(generated) {
                    in_span := false;
                    closed_spans := (closed_spans + 1);
                    phase := "done";
                    break;
                  } else {
                    if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                      generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                      if helpers.EndsWithRightDelimiter(generated) {
                        in_span := false;
                        closed_spans := (closed_spans + 1);
                        phase := "done";
                      } else {
                        in_span := true;
                        phase := "span";
                      }
                    } else {
                      generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
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
    }
    remainingSteps := stepsLeft;
  }

}