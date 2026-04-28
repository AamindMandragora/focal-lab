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
    var closeDelay := 0;
    var sawLeft := false;
    var sawRight := false;
    while ((stepsLeft > 0) && (!sawRight))
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
        if ((stepsLeft <= 4) || (reasoningSteps >= 6)) {
          next_token, new_steps := helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
        } else {
          next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
        }
        generated := (generated + [next_token]);
        stepsLeft := new_steps;
        reasoningSteps := (reasoningSteps + 1);
        if ((next_token == LeftDelimiter) || (next_token == " <<")) {
          sawLeft := true;
          phase := 1;
        }
      } else {
        if ((phase == 1) && (helpers.CanConstrain(generated))) {
          var suffix := helpers.LongestValidSuffix(generated);
          var complete_now := parser.IsCompletePrefix(suffix);
          var continuation_count := parser.ValidContinuationCount(suffix);
          var distance := parser.ParserDistanceToComplete(suffix);
          var next_token: Token;
          var new_steps: nat;
          if ((complete_now) && (closeDelay >= 1) && (((continuation_count <= 1) || (stepsLeft <= (distance + 1)) || (answerSteps >= 4)))) {
            next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
            generated := (generated + [next_token]);
            stepsLeft := new_steps;
            if ((next_token == RightDelimiter) || (next_token == " >>")) {
              sawRight := true;
              phase := 2;
            } else {
              answerSteps := (answerSteps + 1);
              closeDelay := (closeDelay + 1);
            }
          } else {
            next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
            generated := (generated + [next_token]);
            stepsLeft := new_steps;
            if ((next_token == RightDelimiter) || (next_token == " >>")) {
              sawRight := true;
              phase := 2;
            } else {
              answerSteps := (answerSteps + 1);
              suffix := helpers.LongestValidSuffix(generated);
              if parser.IsCompletePrefix(suffix) {
                closeDelay := (closeDelay + 1);
              }
            }
          }
        } else {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}