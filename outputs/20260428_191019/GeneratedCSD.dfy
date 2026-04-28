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
    var verified_spans := 0;
    var reasoning_tokens := 0;
    var close_ready := 0;
    var post_span_tokens := 0;
    var final_span_goal := 0;
    while ((stepsLeft > 0) && (phase < 4))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if phase == 0 {
        var next_token := eosToken;
        var new_steps := stepsLeft;
        if ((verified_spans == 0) && (stepsLeft > 8)) {
          next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
        } else {
          if stepsLeft > 5 {
            next_token, new_steps := helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
          } else {
            next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
          }
        }
        generated := (generated + [next_token]);
        stepsLeft := new_steps;
        if ((next_token == LeftDelimiter) || (next_token == " <<")) {
          phase := 1;
          close_ready := 0;
          final_span_goal := 0;
        } else {
          reasoning_tokens := (reasoning_tokens + 1);
          if ((next_token == ".") || (next_token == ":") || (next_token == ";") || (next_token == "\n")) {
            close_ready := 1;
          } else {
            if ((next_token == "therefore") || (next_token == "Therefore") || (next_token == "total") || (next_token == "Total") || (next_token == "answer") || (next_token == "Answer")) {
              close_ready := 1;
            }
          }
          if ((verified_spans == 0) && (close_ready > 0) && (stepsLeft <= 12)) {
            phase := 0;
          } else {
            if ((verified_spans > 0) && (post_span_tokens > 0) && (close_ready > 0)) {
              phase := 0;
            }
          }
        }
      } else {
        if ((phase == 1) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
          var suffix := helpers.LongestValidSuffix(generated);
          if ((verified_spans == 0) && (helpers.CanExtendConstrained(generated)) && (parser.ValidContinuationCount(suffix) > 0) && (stepsLeft > 4) && (close_ready == 0)) {
            generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
          } else {
            if ((helpers.CanExtendConstrained(generated)) && (parser.ValidContinuationCount(suffix) > 1) && (stepsLeft > (helpers.MinStepsToComplete(generated) + 2)) && (final_span_goal == 0)) {
              generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
            } else {
              var next_token := eosToken;
              var new_steps := stepsLeft;
              next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
              generated := (generated + [next_token]);
              stepsLeft := new_steps;
              if ((next_token == RightDelimiter) || (next_token == " >>")) {
                verified_spans := (verified_spans + 1);
                post_span_tokens := 0;
                close_ready := 0;
                if ((verified_spans >= 2) || (stepsLeft <= 2)) {
                  phase := 3;
                } else {
                  phase := 2;
                }
              } else {
                phase := 1;
              }
            }
          }
        } else {
          if ((phase == 1) && (helpers.CanConstrain(generated)) && (!parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
            if ((verified_spans == 0) && (stepsLeft <= (helpers.MinStepsToComplete(generated) + 3))) {
              generated, stepsLeft := helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft);
            } else {
              generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
            }
          } else {
            if ((phase == 1) && (helpers.CanExtendConstrained(generated))) {
              generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
            } else {
              if phase == 2 {
                var next_token := eosToken;
                var new_steps := stepsLeft;
                if ((verified_spans == 1) && (stepsLeft > 8) && (close_ready == 0)) {
                  next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
                } else {
                  if ((verified_spans == 1) && (stepsLeft > 5)) {
                    next_token, new_steps := helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
                  } else {
                    next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
                  }
                }
                generated := (generated + [next_token]);
                stepsLeft := new_steps;
                if ((next_token == LeftDelimiter) || (next_token == " <<")) {
                  phase := 1;
                  final_span_goal := 1;
                  close_ready := 0;
                } else {
                  post_span_tokens := (post_span_tokens + 1);
                  if ((next_token == ".") || (next_token == ":") || (next_token == ";") || (next_token == "\n")) {
                    close_ready := 1;
                  } else {
                    if ((next_token == "therefore") || (next_token == "Therefore") || (next_token == "total") || (next_token == "Total") || (next_token == "answer") || (next_token == "Answer")) {
                      close_ready := 1;
                    }
                  }
                  if stepsLeft <= 6 {
                    phase := 2;
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