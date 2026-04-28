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
    var close_after_complete := 0;
    var opened_span := 0;
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
        if ((stepsLeft <= 4) || (reason_steps >= 6)) {
          next_token, new_steps := helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
        } else {
          next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
        }
        generated := (generated + [next_token]);
        stepsLeft := new_steps;
        if ((next_token == "<<") || (next_token == " <<")) {
          phase := 1;
          opened_span := 1;
        } else {
          reason_steps := (reason_steps + 1);
        }
      } else {
        if ((phase == 1) && (helpers.CanConstrain(generated))) {
          generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
          answer_steps := (answer_steps + 1);
          if parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)) {
            close_after_complete := 1;
          }
        } else {
          if ((phase == 1) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
            var suffix := helpers.LongestValidSuffix(generated);
            var continuation_count := parser.ValidContinuationCount(suffix);
            var distance := parser.ParserDistanceToComplete(suffix);
            if ((((close_after_complete > 0) && (answer_steps >= 2))) || (continuation_count <= 1) || (stepsLeft <= 2) || (((distance == 0) && (answer_steps >= 4)))) {
              var next_token := eosToken;
              var new_steps := stepsLeft;
              next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
              generated := (generated + [next_token]);
              stepsLeft := new_steps;
              if next_token == ">>" {
                phase := 2;
              } else {
                answer_steps := (answer_steps + 1);
                close_after_complete := (close_after_complete + 1);
              }
            } else {
              if helpers.CanExtendConstrained(generated) {
                generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                answer_steps := (answer_steps + 1);
                close_after_complete := (close_after_complete + 1);
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
                }
              }
            }
          } else {
            if phase == 2 {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              phase := 3;
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