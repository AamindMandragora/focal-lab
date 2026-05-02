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
    var phase := "reasoning";
    var inside_span := false;
    var closed_spans := 0;
    var checkpoint := [];
    var has_checkpoint := false;
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
      }
      if inside_span {
        if helpers.EndsWithRightDelimiter(generated) {
          inside_span := false;
          closed_spans := (closed_spans + 1);
          has_checkpoint := false;
          if phase == "finalizing" {
            final_span_closed := true;
          } else {
            phase := "reasoning";
          }
          break;
        }
        if helpers.IsDead(generated) {
          if has_checkpoint {
            generated := helpers.RestoreCheckpoint(checkpoint);
            inside_span := false;
            has_checkpoint := false;
            phase := "finalizing";
            break;
          }
          break;
        }
        if ((helpers.IsComplete(generated)) && (helpers.ValidContinuationCount(generated) <= 1)) {
          phase := "finalizing";
        }
        if helpers.CanConstrain(generated) {
          if ((!has_checkpoint) && (helpers.ParserDistanceToComplete(generated) <= 2)) {
            checkpoint := helpers.Checkpoint(generated);
            has_checkpoint := true;
          }
          generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
          if helpers.EndsWithRightDelimiter(generated) {
            inside_span := false;
            closed_spans := (closed_spans + 1);
            has_checkpoint := false;
            if phase == "finalizing" {
              final_span_closed := true;
            } else {
              phase := "reasoning";
            }
          }
          break;
        }
        if helpers.IsComplete(generated) {
          break;
        }
        if has_checkpoint {
          generated := helpers.RestoreCheckpoint(checkpoint);
          inside_span := false;
          has_checkpoint := false;
          phase := "finalizing";
          break;
        }
        break;
      }
      if helpers.EndsWithLeftDelimiter(generated) {
        inside_span := true;
        has_checkpoint := false;
        if phase != "finalizing" {
          phase := "scratch";
        }
        break;
      }
      if helpers.EndsWithRightDelimiter(generated) {
        closed_spans := (closed_spans + 1);
        if phase == "finalizing" {
          final_span_closed := true;
        } else {
          phase := "reasoning";
        }
        break;
      }
      if phase == "reasoning" {
        if ((closed_spans > 0) || (helpers.MinStepsToComplete(generated) >= stepsLeft)) {
          phase := "finalizing";
        }
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          has_checkpoint := false;
          if phase != "finalizing" {
            phase := "scratch";
          }
        } else {
          if helpers.EndsWithRightDelimiter(generated) {
            closed_spans := (closed_spans + 1);
          }
        }
        break;
      }
      if phase == "scratch" {
        if ((closed_spans >= 1) || (helpers.MinStepsToComplete(generated) >= stepsLeft)) {
          phase := "finalizing";
        }
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          has_checkpoint := false;
        } else {
          if helpers.EndsWithRightDelimiter(generated) {
            closed_spans := (closed_spans + 1);
            phase := "reasoning";
          }
        }
        break;
      }
      if phase == "finalizing" {
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          has_checkpoint := false;
        } else {
          if helpers.EndsWithRightDelimiter(generated) {
            closed_spans := (closed_spans + 1);
            final_span_closed := true;
          }
        }
        break;
      }
      break;
      if stepsLeft >= stepsLeftBeforeIteration {
        break;
      } else {
        break;
      }
    }
    remainingSteps := stepsLeft;
  }

}