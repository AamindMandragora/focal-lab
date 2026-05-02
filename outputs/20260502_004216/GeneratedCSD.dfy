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
    var has_checkpoint := false;
    var checkpoint := [];
    var used_restore := false;
    var final_span_closed := false;
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
        if helpers.IsDead(generated) {
          if ((has_checkpoint) && (!used_restore)) {
            generated := helpers.RestoreCheckpoint(checkpoint);
            used_restore := true;
            has_checkpoint := false;
            phase := "finalizing";
            in_span := false;
          } else {
            break;
          }
        } else {
          if in_span {
            if helpers.EndsWithRightDelimiter(generated) {
              in_span := false;
              closed_spans := (closed_spans + 1);
              if phase == "finalizing" {
                final_span_closed := true;
              } else {
                phase := "reason";
              }
            } else {
              if helpers.CanConstrain(generated) {
                generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
                if helpers.EndsWithRightDelimiter(generated) {
                  in_span := false;
                  closed_spans := (closed_spans + 1);
                  if phase == "finalizing" {
                    final_span_closed := true;
                  } else {
                    phase := "reason";
                  }
                }
              } else {
                break;
              }
            }
          } else {
            if phase == "reason" {
              if ((!has_checkpoint) && (!used_restore) && (stepsLeft >= 6)) {
                checkpoint := helpers.Checkpoint(generated);
                has_checkpoint := true;
              }
              if closed_spans > 0 {
                phase := "finalizing";
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                if helpers.EndsWithLeftDelimiter(generated) {
                  in_span := true;
                }
              } else {
                if ((stepsLeft <= 4) || (helpers.MinStepsToComplete(generated) >= stepsLeft)) {
                  phase := "finalizing";
                  generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                  if helpers.EndsWithLeftDelimiter(generated) {
                    in_span := true;
                  }
                } else {
                  generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                  if helpers.EndsWithLeftDelimiter(generated) {
                    in_span := true;
                    phase := "finalizing";
                  }
                }
              }
            } else {
              if phase == "finalizing" {
                if ((!has_checkpoint) && (!used_restore)) {
                  checkpoint := helpers.Checkpoint(generated);
                  has_checkpoint := true;
                }
                if helpers.EndsWithLeftDelimiter(generated) {
                  in_span := true;
                } else {
                  generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                  if helpers.EndsWithLeftDelimiter(generated) {
                    in_span := true;
                  }
                }
              } else {
                break;
              }
            }
          }
        }
      }
      if final_span_closed {
        break;
      } else {
        if stepsLeft >= stepsLeftBeforeIteration {
          break;
        }
      }
      if stepsLeft >= stepsLeftBeforeIteration {
        break;
      } else {
        break;
      }
    }
    remainingSteps := stepsLeft;
  }

}