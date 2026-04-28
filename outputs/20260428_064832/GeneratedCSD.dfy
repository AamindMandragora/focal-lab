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
    var phase := 0;
    var reasonCount := 0;
    var spanTokens := 0;
    var milestoneSeen := 0;
    var recentCue := 0;
    var lastToken := "";
    var next_token := eosToken;
    var new_steps := stepsLeft;
    while ((stepsLeft > 0) && (phase < 3))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if phase == 0 {
        var shouldOpen := 0;
        if ((milestoneSeen > 0) && (reasonCount >= 6)) {
          shouldOpen := 1;
        } else {
          if reasonCount >= 18 {
            shouldOpen := 1;
          } else {
            if stepsLeft <= 8 {
              shouldOpen := 1;
            }
          }
        }
        if shouldOpen > 0 {
          generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
          phase := 1;
        } else {
          next_token, new_steps := helpers.UnconstrainedStep(prompt, generated, stepsLeft);
          generated := (generated + [next_token]);
          stepsLeft := new_steps;
          reasonCount := (reasonCount + 1);
          lastToken := next_token;
          if ((next_token == ".") || (next_token == ":") || (next_token == ";") || (next_token == " not ") || (next_token == "?") || (next_token == "NL")) {
            milestoneSeen := 1;
            recentCue := 0;
          } else {
            if ((next_token == "therefore") || (next_token == "Thus") || (next_token == "thus") || (next_token == "so") || (next_token == "total") || (next_token == "Total") || (next_token == "answer") || (next_token == "Answer") || (next_token == "=")) {
              recentCue := 1;
            } else {
              if recentCue > 0 {
                if ((next_token == "is") || (next_token == ":") || (next_token == "=")) {
                  milestoneSeen := 1;
                }
                recentCue := 0;
              }
            }
          }
        }
      } else {
        if ((phase == 1) && (helpers.CanConstrain(generated))) {
          generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
          spanTokens := (spanTokens + 1);
          phase := 2;
        } else {
          if phase == 1 {
            break;
          } else {
            if ((phase == 2) && (helpers.CanConstrain(generated))) {
              generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
              spanTokens := (spanTokens + 1);
            } else {
              if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
                var suffix := helpers.LongestValidSuffix(generated);
                var continuationCount := parser.ValidContinuationCount(suffix);
                var distance := parser.ParserDistanceToComplete(suffix);
                var closeNow := 0;
                if !helpers.CanExtendConstrained(generated) {
                  closeNow := 1;
                } else {
                  if stepsLeft <= (distance + 1) {
                    closeNow := 1;
                  } else {
                    if spanTokens >= 9 {
                      closeNow := 1;
                    } else {
                      if ((continuationCount <= 1) && (spanTokens >= 3)) {
                        closeNow := 1;
                      }
                    }
                  }
                }
                if closeNow > 0 {
                  generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                  phase := 3;
                } else {
                  if helpers.CanExtendConstrained(generated) {
                    generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                    spanTokens := (spanTokens + 1);
                  } else {
                    generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                    phase := 3;
                  }
                }
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