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
    var answer_ready := false;
    var checkpoint := [];
    var has_checkpoint := false;
    var final_span_closed := false;
    var recovered_this_span := false;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      var stepsLeftBeforeIteration := stepsLeft;
      if final_span_closed {
        break;
      } else {
        if inside_span {
          if helpers.EndsWithRightDelimiter(generated) {
            inside_span := false;
            closed_spans := (closed_spans + 1);
            has_checkpoint := false;
            recovered_this_span := false;
            if phase == "final" {
              final_span_closed := true;
              break;
            } else {
              phase := "reason";
            }
          } else {
            if helpers.IsComplete(generated) {
              if !helpers.EndsWithRightDelimiter(generated) {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                if helpers.EndsWithRightDelimiter(generated) {
                  inside_span := false;
                  closed_spans := (closed_spans + 1);
                  has_checkpoint := false;
                  recovered_this_span := false;
                  if phase == "final" {
                    final_span_closed := true;
                    break;
                  } else {
                    phase := "reason";
                  }
                }
              } else {
                inside_span := false;
                closed_spans := (closed_spans + 1);
                has_checkpoint := false;
                recovered_this_span := false;
                if phase == "final" {
                  final_span_closed := true;
                  break;
                } else {
                  phase := "reason";
                }
              }
            } else {
              if helpers.IsDead(generated) {
                if ((has_checkpoint) && (!recovered_this_span)) {
                  generated := helpers.RestoreIfDead(generated, checkpoint);
                  inside_span := false;
                  has_checkpoint := false;
                  recovered_this_span := true;
                  phase := "reason";
                  generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                  if helpers.EndsWithLeftDelimiter(generated) {
                    inside_span := true;
                    has_checkpoint := false;
                    recovered_this_span := false;
                  }
                } else {
                  inside_span := false;
                  has_checkpoint := false;
                  phase := "reason";
                  generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                  if helpers.EndsWithLeftDelimiter(generated) {
                    inside_span := true;
                    has_checkpoint := false;
                    recovered_this_span := false;
                  }
                }
              } else {
                if !has_checkpoint {
                  if helpers.CanConstrain(generated) {
                    checkpoint := helpers.Checkpoint(generated);
                    has_checkpoint := true;
                    generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
                    if helpers.EndsWithRightDelimiter(generated) {
                      inside_span := false;
                      closed_spans := (closed_spans + 1);
                      has_checkpoint := false;
                      recovered_this_span := false;
                      if phase == "final" {
                        final_span_closed := true;
                        break;
                      } else {
                        phase := "reason";
                      }
                    }
                  } else {
                    inside_span := false;
                    has_checkpoint := false;
                    phase := "reason";
                    generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                    if helpers.EndsWithLeftDelimiter(generated) {
                      inside_span := true;
                      has_checkpoint := false;
                      recovered_this_span := false;
                    }
                  }
                } else {
                  if helpers.CanConstrain(generated) {
                    generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
                    if helpers.EndsWithRightDelimiter(generated) {
                      inside_span := false;
                      closed_spans := (closed_spans + 1);
                      has_checkpoint := false;
                      recovered_this_span := false;
                      if phase == "final" {
                        final_span_closed := true;
                        break;
                      } else {
                        phase := "reason";
                      }
                    }
                  } else {
                    inside_span := false;
                    has_checkpoint := false;
                    phase := "reason";
                    generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                    if helpers.EndsWithLeftDelimiter(generated) {
                      inside_span := true;
                      has_checkpoint := false;
                      recovered_this_span := false;
                    }
                  }
                }
              }
            }
          }
        } else {
          if closed_spans > 0 {
            answer_ready := true;
          } else {
            if stepsLeft <= 8 {
              answer_ready := true;
            } else {
              if helpers.MinStepsToComplete(generated) >= stepsLeft {
                answer_ready := true;
              } else {
                if helpers.ParserDistanceToComplete(generated) >= stepsLeft {
                  answer_ready := true;
                } else {
                  if helpers.ValidContinuationCount(generated) <= 1 {
                    answer_ready := true;
                  } else {
                    answer_ready := false;
                  }
                }
              }
            }
          }
          if answer_ready {
            phase := "final";
          } else {
            phase := "reason";
          }
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        }
      }
      if stepsLeft >= stepsLeftBeforeIteration {
        break;
      }
    }
    remainingSteps := stepsLeft;
  }

}