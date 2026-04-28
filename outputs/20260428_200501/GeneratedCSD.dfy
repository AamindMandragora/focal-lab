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
    var closed_spans := 0;
    var reason_tokens := 0;
    var milestones := 0;
    var summary_cues := 0;
    var final_ready := 0;
    var next_token := eosToken;
    var new_steps := stepsLeft;
    var suffix := [];
    var distance := 0;
    var continuations := 0;
    while ((stepsLeft > 0) && (phase < 3) && (closed_spans < 2))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if phase == 0 {
        if ((final_ready > 0) || (!helpers.HasBudget(stepsLeft, 8))) {
          next_token, new_steps := helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
        } else {
          if ((milestones >= 2) && (reason_tokens >= 6)) {
            next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
          } else {
            if ((summary_cues >= 1) && (reason_tokens >= 5)) {
              next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
            } else {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              reason_tokens := (reason_tokens + 1);
              if |generated| > 0 {
                next_token := generated[(|generated| - 1)];
                if ((next_token == ".") || (next_token == ",") || (next_token == ":") || (next_token == ";") || (next_token == "\n")) {
                  milestones := (milestones + 1);
                }
                if ((next_token == "therefore") || (next_token == "Thus") || (next_token == "thus") || (next_token == "so") || (next_token == "total") || (next_token == "answer") || (next_token == "=")) {
                  summary_cues := (summary_cues + 1);
                  milestones := (milestones + 1);
                }
                if ((reason_tokens >= 10) && (milestones >= 1)) {
                  final_ready := 1;
                }
                if reason_tokens >= 14 {
                  final_ready := 1;
                }
              }
              continue;
            }
          }
        }
        generated := (generated + [next_token]);
        stepsLeft := new_steps;
        if ((next_token == LeftDelimiter) || (next_token == " <<")) {
          phase := 1;
        } else {
          reason_tokens := (reason_tokens + 1);
          if ((next_token == ".") || (next_token == ",") || (next_token == ":") || (next_token == ";") || (next_token == "\n")) {
            milestones := (milestones + 1);
          }
          if ((next_token == "therefore") || (next_token == "Thus") || (next_token == "thus") || (next_token == "so") || (next_token == "total") || (next_token == "answer") || (next_token == "=")) {
            summary_cues := (summary_cues + 1);
            milestones := (milestones + 1);
          }
          if ((reason_tokens >= 10) && (milestones >= 1)) {
            final_ready := 1;
          }
          if reason_tokens >= 14 {
            final_ready := 1;
          }
        }
      } else {
        if ((phase == 1) && (helpers.CanConstrain(generated))) {
          suffix := helpers.LongestValidSuffix(generated);
          distance := parser.ParserDistanceToComplete(suffix);
          if distance <= 1 {
            generated, stepsLeft := helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft);
          } else {
            generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
          }
        } else {
          if ((phase == 1) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
            suffix := helpers.LongestValidSuffix(generated);
            continuations := parser.ValidContinuationCount(suffix);
            distance := parser.ParserDistanceToComplete(suffix);
            next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
            generated := (generated + [next_token]);
            stepsLeft := new_steps;
            if ((next_token == RightDelimiter) || (next_token == " >>")) {
              closed_spans := (closed_spans + 1);
              phase := 3;
            } else {
              if ((continuations <= 1) || (!helpers.HasBudget(stepsLeft, 3))) {
                final_ready := 1;
              }
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