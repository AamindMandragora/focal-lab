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
    var wrap_steps := 0;
    var answer_steps := 0;
    var nudge_steps := 0;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if inside_span {
        if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
          generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
          if helpers.EndsWithRightDelimiter(generated) {
            inside_span := false;
            closed_spans := (closed_spans + 1);
            phase := "done";
          }
        } else {
          break;
        }
      } else {
        if closed_spans > 0 {
          break;
        } else {
          if phase == "reason" {
            if ((reason_steps < 56) && (helpers.HasBudget(stepsLeft, 12))) {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              reason_steps := (reason_steps + 1);
              if helpers.EndsWithLeftDelimiter(generated) {
                inside_span := true;
                phase := "span";
              } else {
                if reason_steps >= 56 {
                  phase := "wrap";
                } else {
                  if ((reason_steps >= 44) && (!helpers.HasBudget(stepsLeft, 10))) {
                    phase := "wrap";
                  }
                }
              }
            } else {
              phase := "wrap";
              if helpers.HasBudget(stepsLeft, 1) {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                wrap_steps := (wrap_steps + 1);
                if helpers.EndsWithLeftDelimiter(generated) {
                  inside_span := true;
                  phase := "span";
                }
              } else {
                break;
              }
            }
          } else {
            if phase == "wrap" {
              if ((wrap_steps < 8) && (helpers.HasBudget(stepsLeft, 8))) {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                wrap_steps := (wrap_steps + 1);
                if helpers.EndsWithLeftDelimiter(generated) {
                  inside_span := true;
                  phase := "span";
                } else {
                  if wrap_steps >= 8 {
                    phase := "answer";
                  } else {
                    if ((wrap_steps >= 4) && (!helpers.HasBudget(stepsLeft, 7))) {
                      phase := "answer";
                    }
                  }
                }
              } else {
                phase := "answer";
                if helpers.HasBudget(stepsLeft, 1) {
                  generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                  answer_steps := (answer_steps + 1);
                  if helpers.EndsWithLeftDelimiter(generated) {
                    inside_span := true;
                    phase := "span";
                  }
                } else {
                  break;
                }
              }
            } else {
              if phase == "answer" {
                if ((answer_steps < 10) && (helpers.HasBudget(stepsLeft, 5))) {
                  if answer_steps < 6 {
                    generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                  } else {
                    generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                  }
                  answer_steps := (answer_steps + 1);
                  if helpers.EndsWithLeftDelimiter(generated) {
                    inside_span := true;
                    phase := "span";
                  } else {
                    if answer_steps >= 10 {
                      phase := "nudge";
                    }
                  }
                } else {
                  phase := "nudge";
                  if helpers.HasBudget(stepsLeft, 1) {
                    generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                    nudge_steps := (nudge_steps + 1);
                    if helpers.EndsWithLeftDelimiter(generated) {
                      inside_span := true;
                      phase := "span";
                    }
                  } else {
                    break;
                  }
                }
              } else {
                if phase == "nudge" {
                  if helpers.HasBudget(stepsLeft, 1) {
                    generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                    nudge_steps := (nudge_steps + 1);
                    if helpers.EndsWithLeftDelimiter(generated) {
                      inside_span := true;
                      phase := "span";
                    }
                  } else {
                    break;
                  }
                } else {
                  if phase == "span" {
                    if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
                      generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                      if helpers.EndsWithRightDelimiter(generated) {
                        inside_span := false;
                        closed_spans := (closed_spans + 1);
                        phase := "done";
                      }
                    } else {
                      break;
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
    }
    remainingSteps := stepsLeft;
  }

}