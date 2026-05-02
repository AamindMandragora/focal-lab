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
    var scratch_spans := 0;
    var nudge_mode := false;
    var has_checkpoint := false;
    var checkpoint := [];
    while ((stepsLeft > 0) && (((closed_spans == 0) || (scratch_spans == closed_spans))))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      var stepsLeftBeforeIteration := stepsLeft;
      if in_span {
        if helpers.IsDead(generated) {
          if has_checkpoint {
            generated := helpers.RestoreCheckpoint(checkpoint);
            in_span := false;
            nudge_mode := true;
            phase := "final";
            has_checkpoint := false;
            scratch_spans := closed_spans;
            break;
          } else {
            break;
          }
        } else {
          if helpers.CanConstrain(generated) {
            if !has_checkpoint {
              checkpoint := helpers.Checkpoint(generated);
              has_checkpoint := true;
            }
            generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
            if helpers.EndsWithRightDelimiter(generated) {
              in_span := false;
              nudge_mode := false;
              closed_spans := (closed_spans + 1);
              has_checkpoint := false;
              if phase == "scratch" {
                scratch_spans := closed_spans;
                phase := "reason";
              } else {
                phase := "final_done";
                break;
              }
            } else {
              if helpers.IsDead(generated) {
                if has_checkpoint {
                  generated := helpers.RestoreCheckpoint(checkpoint);
                  in_span := false;
                  nudge_mode := true;
                  phase := "final";
                  has_checkpoint := false;
                  scratch_spans := closed_spans;
                } else {
                  break;
                }
              }
            }
          } else {
            break;
          }
        }
      } else {
        if helpers.IsDead(generated) {
          if has_checkpoint {
            generated := helpers.RestoreCheckpoint(checkpoint);
            has_checkpoint := false;
            nudge_mode := true;
            phase := "final";
            scratch_spans := closed_spans;
          } else {
            break;
          }
        } else {
          if nudge_mode {
            generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
            if helpers.EndsWithLeftDelimiter(generated) {
              in_span := true;
              nudge_mode := false;
              checkpoint := helpers.Checkpoint(generated);
              has_checkpoint := true;
            }
          } else {
            var need_final := closed_spans == 0;
            var budget_pressure := stepsLeft <= (4 + helpers.MinStepsToComplete(generated));
            var near_complete := helpers.ParserDistanceToComplete(generated) <= 2;
            var many_choices := helpers.ValidContinuationCount(generated) > 1;
            if ((need_final) && (((phase == "final") || (budget_pressure) || (near_complete)))) {
              phase := "final";
              nudge_mode := true;
              generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
              if helpers.EndsWithLeftDelimiter(generated) {
                in_span := true;
                nudge_mode := false;
                checkpoint := helpers.Checkpoint(generated);
                has_checkpoint := true;
              }
            } else {
              if ((closed_spans == 0) && (!budget_pressure) && (!near_complete) && (many_choices)) {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                if helpers.EndsWithLeftDelimiter(generated) {
                  in_span := true;
                  nudge_mode := false;
                  phase := "scratch";
                  checkpoint := helpers.Checkpoint(generated);
                  has_checkpoint := true;
                }
              } else {
                generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                if helpers.EndsWithLeftDelimiter(generated) {
                  in_span := true;
                  nudge_mode := false;
                  if ((closed_spans == 0) && (!budget_pressure) && (!near_complete)) {
                    phase := "scratch";
                  } else {
                    phase := "final";
                  }
                  checkpoint := helpers.Checkpoint(generated);
                  has_checkpoint := true;
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