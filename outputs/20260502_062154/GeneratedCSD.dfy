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
    var reason_steps := 0;
    var open_attempts := 0;
    var answer_seek_steps := 0;
    var setup_ready := false;
    var late_ready := false;
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
        if inside_span {
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
                phase := "span";
              }
            } else {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              if helpers.EndsWithRightDelimiter(generated) {
                inside_span := false;
                closed_spans := (closed_spans + 1);
                phase := "done";
                break;
              } else {
                if helpers.EndsWithLeftDelimiter(generated) {
                  inside_span := true;
                  phase := "span";
                } else {
                  phase := "span";
                }
              }
            }
          }
        } else {
          if phase == "open" {
            var should_nudge := false;
            if open_attempts < 3 {
              should_nudge := true;
            } else {
              if answer_seek_steps < 6 {
                should_nudge := true;
              } else {
                if ((late_ready) && (open_attempts < 8)) {
                  should_nudge := true;
                } else {
                  if ((setup_ready) && (helpers.IsComplete(generated))) {
                    should_nudge := true;
                  } else {
                    if ((setup_ready) && (helpers.MinStepsToComplete(generated) <= 2)) {
                      should_nudge := true;
                    } else {
                      if ((setup_ready) && (helpers.ParserDistanceToComplete(generated) <= 2)) {
                        should_nudge := true;
                      } else {
                        if ((setup_ready) && (helpers.ValidContinuationCount(generated) <= 3)) {
                          should_nudge := true;
                        }
                      }
                    }
                  }
                }
              }
            }
            if should_nudge {
              generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
              open_attempts := (open_attempts + 1);
              answer_seek_steps := (answer_seek_steps + 1);
              if helpers.EndsWithLeftDelimiter(generated) {
                inside_span := true;
                phase := "span";
                open_attempts := 0;
              } else {
                phase := "open";
              }
            } else {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              reason_steps := (reason_steps + 1);
              answer_seek_steps := (answer_seek_steps + 1);
              if reason_steps >= 24 {
                setup_ready := true;
              }
              if reason_steps >= 32 {
                late_ready := true;
              }
              if helpers.EndsWithLeftDelimiter(generated) {
                inside_span := true;
                phase := "span";
                open_attempts := 0;
              } else {
                phase := "answer_seek";
              }
            }
          } else {
            if phase == "done" {
              break;
            } else {
              var should_open := false;
              if reason_steps >= 24 {
                setup_ready := true;
              }
              if reason_steps >= 32 {
                late_ready := true;
              }
              if setup_ready {
                if reason_steps >= 28 {
                  should_open := true;
                } else {
                  if helpers.IsComplete(generated) {
                    should_open := true;
                  } else {
                    if ((late_ready) && (helpers.MinStepsToComplete(generated) <= 2)) {
                      should_open := true;
                    } else {
                      if ((late_ready) && (helpers.ParserDistanceToComplete(generated) <= 2)) {
                        should_open := true;
                      } else {
                        if ((late_ready) && (helpers.ValidContinuationCount(generated) <= 3)) {
                          should_open := true;
                        }
                      }
                    }
                  }
                }
              }
              if should_open {
                generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                open_attempts := 1;
                answer_seek_steps := 1;
                if helpers.EndsWithLeftDelimiter(generated) {
                  inside_span := true;
                  phase := "span";
                  open_attempts := 0;
                } else {
                  phase := "open";
                }
              } else {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                reason_steps := (reason_steps + 1);
                if reason_steps >= 24 {
                  setup_ready := true;
                }
                if reason_steps >= 32 {
                  late_ready := true;
                }
                if helpers.EndsWithLeftDelimiter(generated) {
                  inside_span := true;
                  phase := "span";
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