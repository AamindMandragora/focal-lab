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
    var final_phase := false;
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
      if helpers.IsDead(generated) {
        if has_checkpoint {
          generated := helpers.RestoreCheckpoint(checkpoint);
          has_checkpoint := false;
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          if helpers.EndsWithLeftDelimiter(generated) {
            inside_span := true;
            phase := "span";
          } else {
            if helpers.EndsWithRightDelimiter(generated) {
              inside_span := false;
              closed_spans := (closed_spans + 1);
              phase := "reason";
            } else {
              phase := (if !inside_span then "reason" else "span");
              break;
            }
          }
          break;
        }
        break;
      }
      if inside_span {
        if helpers.EndsWithRightDelimiter(generated) {
          inside_span := false;
          closed_spans := (closed_spans + 1);
          phase := "after_span";
          if final_phase {
            break;
          }
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          if helpers.EndsWithLeftDelimiter(generated) {
            inside_span := true;
            phase := "span";
          } else {
            if helpers.EndsWithRightDelimiter(generated) {
              inside_span := false;
              closed_spans := (closed_spans + 1);
              phase := "after_span";
            } else {
              phase := "reason";
              break;
            }
          }
          break;
        }
        if helpers.CanConstrain(generated) {
          if !has_checkpoint {
            checkpoint := helpers.Checkpoint(generated);
            has_checkpoint := true;
          }
          generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
          if helpers.EndsWithRightDelimiter(generated) {
            inside_span := false;
            closed_spans := (closed_spans + 1);
            phase := "after_span";
            if final_phase {
              break;
            }
          } else {
            phase := "span";
            break;
          }
          break;
        }
        break;
      }
      if phase == "after_span" {
        if ((final_phase) && (closed_spans > 0)) {
          break;
        }
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          phase := "span";
          has_checkpoint := false;
        } else {
          if helpers.EndsWithRightDelimiter(generated) {
            inside_span := false;
            closed_spans := (closed_spans + 1);
            phase := "after_span";
          } else {
            phase := "reason";
            break;
          }
        }
        break;
      }
      var answer_pressure := false;
      if !helpers.HasBudget(stepsLeft, 6) {
        answer_pressure := true;
      } else {
        if ((closed_spans == 0) && (helpers.HasBudget(stepsLeft, 1)) && (helpers.MinStepsToComplete(generated) <= 1)) {
          answer_pressure := true;
        } else {
          if stall_count >= 4 {
            answer_pressure := true;
          }
        }
      }
      if answer_pressure {
        final_phase := true;
        phase := "seek_span";
      } else {
        if phase == "seek_span" {
          final_phase := true;
        }
      }
      if phase == "seek_span" {
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          phase := "span";
          has_checkpoint := false;
        } else {
          if helpers.EndsWithRightDelimiter(generated) {
            inside_span := false;
            closed_spans := (closed_spans + 1);
            phase := "after_span";
          } else {
            phase := "seek_span";
            break;
          }
        }
        stall_count := (stall_count + 1);
        break;
      }
      generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
      if helpers.EndsWithLeftDelimiter(generated) {
        inside_span := true;
        phase := "span";
        has_checkpoint := false;
        stall_count := 0;
      } else {
        if helpers.EndsWithRightDelimiter(generated) {
          inside_span := false;
          closed_spans := (closed_spans + 1);
          phase := "after_span";
          stall_count := 0;
        } else {
          phase := "reason";
          stall_count := (stall_count + 1);
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