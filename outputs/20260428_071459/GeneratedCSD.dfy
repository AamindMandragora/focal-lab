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
    var reasoningSteps := 0;
    var answerSteps := 0;
    var closeBias := 0;
    var sawLeft := 0;
    var sawRight := 0;
    while ((stepsLeft > 0) && (sawRight == 0))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if phase == 0 {
        var next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
        generated := (generated + [next_token]);
        stepsLeft := new_steps;
        reasoningSteps := (reasoningSteps + 1);
        if next_token == "<<" {
          sawLeft := 1;
          phase := 2;
        } else {
          if next_token == eosToken {
            break;
          } else {
            if ((reasoningSteps >= 6) && (stepsLeft > 0)) {
              generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
              sawLeft := 1;
              phase := 2;
            } else {
              phase := 0;
            }
          }
        }
      } else {
        if ((phase == 2) && (helpers.CanConstrain(generated))) {
          generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
          answerSteps := (answerSteps + 1);
          if answerSteps >= 2 {
            closeBias := 1;
          } else {
            closeBias := 0;
          }
          phase := 2;
        } else {
          if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
            var suffix := helpers.LongestValidSuffix(generated);
            var continuation_count := parser.ValidContinuationCount(suffix);
            var distance := parser.ParserDistanceToComplete(suffix);
            var next_token := eosToken;
            var new_steps := stepsLeft;
            if ((((closeBias > 0) && (continuation_count <= 1))) || (stepsLeft <= 2) || (answerSteps >= 5) || (distance == 0)) {
              next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
              generated := (generated + [next_token]);
              stepsLeft := new_steps;
              if next_token == ">>" {
                sawRight := 1;
                phase := 3;
              } else {
                answerSteps := (answerSteps + 1);
                phase := 2;
              }
            } else {
              if ((phase == 2) && (helpers.CanExtendConstrained(generated))) {
                generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                answerSteps := (answerSteps + 1);
                closeBias := (closeBias + 1);
                phase := 2;
              } else {
                next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                generated := (generated + [next_token]);
                stepsLeft := new_steps;
                if next_token == ">>" {
                  sawRight := 1;
                  phase := 3;
                } else {
                  answerSteps := (answerSteps + 1);
                  phase := 2;
                }
              }
            }
          } else {
            if phase == 2 {
              break;
            } else {
              break;
            }
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}