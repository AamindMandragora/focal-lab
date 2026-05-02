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
    var nudge_steps := 0;
    var final_ready := false;
    while ((stepsLeft > 0) && (closed_spans < 2))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      var stepsLeftBeforeIteration := stepsLeft;
      if helpers.IsDead(generated) {
        break;
      } else {
        if in_span {
          if helpers.EndsWithRightDelimiter(generated) {
            in_span := false;
            closed_spans := (closed_spans + 1);
            if closed_spans >= 2 {
              phase := "done";
              break;
            } else {
              phase := "reason";
              final_ready := true;
              nudge_steps := 0;
            }
          } else {
            if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
              generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
              if helpers.EndsWithRightDelimiter(generated) {
                in_span := false;
                closed_spans := (closed_spans + 1);
                if closed_spans >= 2 {
                  phase := "done";
                  break;
                } else {
                  phase := "reason";
                  final_ready := true;
                  nudge_steps := 0;
                }
              }
            } else {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
            }
          }
        } else {
          if ((phase == "done") || (closed_spans >= 2)) {
            break;
          } else {
            if helpers.EndsWithLeftDelimiter(generated) {
              in_span := true;
              phase := "span";
              if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                if helpers.EndsWithRightDelimiter(generated) {
                  in_span := false;
                  closed_spans := (closed_spans + 1);
                  if closed_spans >= 2 {
                    phase := "done";
                    break;
                  } else {
                    phase := "reason";
                    final_ready := true;
                    nudge_steps := 0;
                  }
                }
              } else {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              }
            } else {
              var near_complete := false;
              var tight_completion := false;
              var low_branching := false;
              if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                if helpers.ParserDistanceToComplete(generated) <= 4 {
                  near_complete := true;
                }
                if helpers.MinStepsToComplete(generated) <= 4 {
                  tight_completion := true;
                }
                if helpers.ValidContinuationCount(generated) <= 5 {
                  low_branching := true;
                }
              }
              var enough_reasoning := reason_steps >= 10;
              var substantial_reasoning := reason_steps >= 6;
              var budget_window := stepsLeft <= 12;
              var budget_pressure := stepsLeft <= 8;
              var prior_span_exists := closed_spans >= 1;
              if !final_ready {
                if prior_span_exists {
                  final_ready := true;
                } else {
                  if ((substantial_reasoning) && (((near_complete) || (tight_completion) || (low_branching)))) {
                    final_ready := true;
                  } else {
                    if ((enough_reasoning) && (budget_window)) {
                      final_ready := true;
                    } else {
                      if reason_steps >= 14 {
                        final_ready := true;
                      } else {
                        if ((budget_pressure) && (reason_steps >= 8)) {
                          final_ready := true;
                        }
                      }
                    }
                  }
                }
              }
              if final_ready {
                generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                nudge_steps := (nudge_steps + 1);
                if helpers.EndsWithLeftDelimiter(generated) {
                  in_span := true;
                  phase := "span";
                } else {
                  if ((nudge_steps >= 4) && (stepsLeft > 0)) {
                    generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                  }
                }
              } else {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                reason_steps := (reason_steps + 1);
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