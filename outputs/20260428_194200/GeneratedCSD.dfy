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
    var spanTokens := 0;
    var closePreference := 0;
    var openedNaturally := 0;
    while ((stepsLeft > 0) && (phase < 3))
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
        if ((stepsLeft <= 3) || (reasoningSteps >= 6)) {
          next_token, new_steps := helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
        } else {
          next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
        }
        generated := (generated + [next_token]);
        stepsLeft := new_steps;
        reasoningSteps := (reasoningSteps + 1);
        if ((next_token == LeftDelimiter) || (next_token == " <<")) {
          phase := 1;
          openedNaturally := 1;
          spanTokens := 0;
          closePreference := 0;
        }
      } else {
        if ((phase == 1) && (helpers.CanConstrain(generated))) {
          var suffix := helpers.LongestValidSuffix(generated);
          var complete_now := parser.IsCompletePrefix(suffix);
          var continuation_count := parser.ValidContinuationCount(suffix);
          var distance := parser.ParserDistanceToComplete(suffix);
          var next_token := eosToken;
          var new_steps := stepsLeft;
          next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
          generated := (generated + [next_token]);
          stepsLeft := new_steps;
          if ((next_token == RightDelimiter) || (next_token == " >>")) {
            phase := 2;
          } else {
            spanTokens := (spanTokens + 1);
            if complete_now {
              closePreference := (closePreference + 1);
            } else {
              if distance <= 1 {
                closePreference := (closePreference + 1);
              } else {
                closePreference := 0;
              }
            }
            if ((continuation_count <= 1) && (spanTokens >= 2)) {
              closePreference := (closePreference + 1);
            }
          }
        } else {
          if ((phase == 1) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
            var next_token := eosToken;
            var new_steps := stepsLeft;
            next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
            generated := (generated + [next_token]);
            stepsLeft := new_steps;
            if ((next_token == RightDelimiter) || (next_token == " >>")) {
              phase := 2;
            } else {
              spanTokens := (spanTokens + 1);
              closePreference := (closePreference + 1);
            }
          } else {
            break;
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}