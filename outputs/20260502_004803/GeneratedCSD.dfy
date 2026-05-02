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
    var has_checkpoint := false;
    var checkpoint := [];
    var stall_count := 0;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      var stepsLeftBeforeIteration := stepsLeft;
      if inside_span {
        if helpers.EndsWithRightDelimiter(generated) {
          inside_span := false;
          closed_spans := (closed_spans + 1);
          phase := "post_span";
          has_checkpoint := false;
          stall_count := 0;
          if ((closed_spans >= 1) && (!helpers.HasBudget(stepsLeft, 2))) {
            break;
          }
        } else {
          if helpers.IsDead(generated) {
            if has_checkpoint {
              generated := helpers.RestoreIfDead(generated, checkpoint);
              inside_span := false;
              phase := "reason";
              has_checkpoint := false;
              stall_count := 0;
            } else {
              break;
            }
          } else {
            if helpers.CanConstrain(generated) {
              if ((!has_checkpoint) && (helpers.HasBudget(stepsLeft, 2))) {
                checkpoint := helpers.Checkpoint(generated);
                has_checkpoint := true;
              }
              var prev_len := |generated|;
              generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
              if |generated| == prev_len {
                stall_count := (stall_count + 1);
              } else {
                stall_count := 0;
              }
              if helpers.EndsWithRightDelimiter(generated) {
                inside_span := false;
                closed_spans := (closed_spans + 1);
                phase := "post_span";
                has_checkpoint := false;
                stall_count := 0;
              } else {
                if ((stall_count >= 2) && (helpers.IsComplete(generated))) {
                  break;
                }
              }
            } else {
              break;
            }
          }
        }
      } else {
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          phase := "span";
          stall_count := 0;
          if !has_checkpoint {
            checkpoint := helpers.Checkpoint(generated);
            has_checkpoint := true;
          }
        } else {
          if helpers.IsDead(generated) {
            if has_checkpoint {
              generated := helpers.RestoreIfDead(generated, checkpoint);
              inside_span := false;
              phase := "reason";
              has_checkpoint := false;
              stall_count := 0;
            } else {
              break;
            }
          } else {
            if closed_spans == 0 {
              if !helpers.HasBudget(stepsLeft, 6) {
                answer_ready := true;
              } else {
                if ((helpers.HasBudget(stepsLeft, 3)) && (helpers.MinStepsToComplete(generated) <= 2)) {
                  answer_ready := true;
                }
              }
            } else {
              if !helpers.HasBudget(stepsLeft, 4) {
                break;
              }
            }
            var prev_len := |generated|;
            generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
            if |generated| == prev_len {
              stall_count := (stall_count + 1);
            } else {
              stall_count := 0;
            }
            if helpers.EndsWithLeftDelimiter(generated) {
              inside_span := true;
              phase := "span";
              if !has_checkpoint {
                checkpoint := helpers.Checkpoint(generated);
                has_checkpoint := true;
              }
              stall_count := 0;
            } else {
              if answer_ready {
                phase := "seek_span";
              } else {
                phase := "reason";
              }
            }
            if stall_count >= 2 {
              break;
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