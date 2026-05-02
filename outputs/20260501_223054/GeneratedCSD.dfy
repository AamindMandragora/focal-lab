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
    var final_span_closed := false;
    var checkpoint := [];
    var has_checkpoint := false;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if in_span {
        if helpers.EndsWithRightDelimiter(generated) {
          in_span := false;
          closed_spans := (closed_spans + 1);
          checkpoint := [];
          has_checkpoint := false;
          if phase == "answer" {
            final_span_closed := true;
            break;
          }
          phase := "reason";
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        } else {
          if helpers.IsComplete(generated) {
            checkpoint := helpers.Checkpoint(generated);
            has_checkpoint := true;
            if helpers.ValidContinuationCount(generated) > 0 {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              if helpers.EndsWithRightDelimiter(generated) {
                in_span := false;
                closed_spans := (closed_spans + 1);
                checkpoint := [];
                has_checkpoint := false;
                if phase == "answer" {
                  final_span_closed := true;
                  break;
                }
                phase := "reason";
              } else {
                generated := helpers.RestoreIfDead(generated, checkpoint);
              }
            } else {
              break;
            }
          } else {
            if helpers.CanConstrain(generated) {
              generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
              if helpers.IsDead(generated) {
                if has_checkpoint {
                  generated := helpers.RestoreCheckpoint(checkpoint);
                  has_checkpoint := false;
                  checkpoint := [];
                } else {
                  break;
                }
              }
            } else {
              break;
            }
          }
        }
      } else {
        if final_span_closed {
          break;
        } else {
          if phase == "answer" {
            generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
            if helpers.EndsWithLeftDelimiter(generated) {
              in_span := true;
              checkpoint := [];
              has_checkpoint := false;
            }
          } else {
            if ((closed_spans == 0) && ((helpers.MinStepsToComplete(generated) + 2) < stepsLeft)) {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              if helpers.EndsWithLeftDelimiter(generated) {
                in_span := true;
                checkpoint := [];
                has_checkpoint := false;
              } else {
                if helpers.EndsWithRightDelimiter(generated) {
                  closed_spans := (closed_spans + 1);
                }
              }
            } else {
              phase := "answer";
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              if helpers.EndsWithLeftDelimiter(generated) {
                in_span := true;
                checkpoint := [];
                has_checkpoint := false;
              }
            }
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}