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
    var reason_steps := 0;
    var answer_steps := 0;
    var close_score := 0;
    var saw_left := 0;
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
        if ((stepsLeft <= 3) || (reason_steps >= 4)) {
          next_token, new_steps := helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
        } else {
          next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
        }
        generated := (generated + [next_token]);
        stepsLeft := new_steps;
        reason_steps := (reason_steps + 1);
        if ((next_token == "<<") || (next_token == " <<")) {
          phase := 1;
          saw_left := 1;
        } else {
          phase := 0;
        }
      } else {
        if ((phase == 1) && (helpers.CanConstrain(generated))) {
          generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
          answer_steps := (answer_steps + 1);
          phase := 1;
        } else {
          if ((phase == 1) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
            var suffix := helpers.LongestValidSuffix(generated);
            var conts := parser.ValidContinuationCount(suffix);
            var dist := parser.ParserDistanceToComplete(suffix);
            if answer_steps >= 3 {
              close_score := (close_score + 1);
            }
            if conts <= 1 {
              close_score := (close_score + 1);
            }
            if stepsLeft <= 2 {
              close_score := (close_score + 1);
            }
            if close_score >= 2 {
              var next_token := eosToken;
              var new_steps := stepsLeft;
              next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
              generated := (generated + [next_token]);
              stepsLeft := new_steps;
              if next_token == ">>" {
                phase := 2;
              } else {
                answer_steps := (answer_steps + 1);
                phase := 1;
              }
            } else {
              if helpers.CanExtendConstrained(generated) {
                generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                answer_steps := (answer_steps + 1);
                phase := 1;
              } else {
                var next_token := eosToken;
                var new_steps := stepsLeft;
                next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
                generated := (generated + [next_token]);
                stepsLeft := new_steps;
                if next_token == ">>" {
                  phase := 2;
                } else {
                  answer_steps := (answer_steps + 1);
                  phase := 1;
                }
              }
            }
          } else {
            generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
            phase := 2;
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}