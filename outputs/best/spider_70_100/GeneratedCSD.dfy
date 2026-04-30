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
    var checkpoint := helpers.Checkpoint(generated);
    var attempts := 0;
    var max_attempts := 2;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if helpers.EndsWithRightDelimiter(generated) {
        break;
      } else {
        if !helpers.ContainsLeftDelimiter(generated) {
          generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
          checkpoint := helpers.Checkpoint(generated);
        } else {
          if helpers.IsDead(generated) {
            if attempts < max_attempts {
              generated := helpers.RestoreCheckpoint(checkpoint);
              attempts := (attempts + 1);
              if helpers.EndsWithRightDelimiter(generated) {
                break;
              } else {
                if helpers.IsComplete(generated) {
                  generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                } else {
                  if helpers.CanConstrain(generated) {
                    generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
                    generated := helpers.RestoreIfDead(generated, checkpoint);
                    if !helpers.IsDead(generated) {
                      checkpoint := helpers.Checkpoint(generated);
                    }
                  } else {
                    break;
                  }
                }
              }
            } else {
              break;
            }
          } else {
            if helpers.CanConstrain(generated) {
              generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
              generated := helpers.RestoreIfDead(generated, checkpoint);
              if helpers.EndsWithRightDelimiter(generated) {
                break;
              } else {
                if helpers.IsDead(generated) {
                  if attempts < max_attempts {
                    generated := helpers.RestoreCheckpoint(checkpoint);
                    attempts := (attempts + 1);
                    if helpers.EndsWithRightDelimiter(generated) {
                      break;
                    } else {
                      if helpers.IsComplete(generated) {
                        generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                      } else {
                        if helpers.CanConstrain(generated) {
                          generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
                          generated := helpers.RestoreIfDead(generated, checkpoint);
                          if !helpers.IsDead(generated) {
                            checkpoint := helpers.Checkpoint(generated);
                          }
                        } else {
                          break;
                        }
                      }
                    }
                  } else {
                    break;
                  }
                } else {
                  checkpoint := helpers.Checkpoint(generated);
                }
              }
            } else {
              if helpers.IsComplete(generated) {
                generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
              } else {
                break;
              }
            }
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}