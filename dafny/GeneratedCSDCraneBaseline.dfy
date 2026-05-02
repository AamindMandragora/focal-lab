include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(
    lm: LM,
    parser: Parser,
    prompt: Prefix,
    generatedPrefix: Prefix,
    insideConstrained: bool,
    currentConstrained: Prefix,
    maxSteps: nat,
    eosToken: Token
  ) returns (
    generated: Prefix,
    insideConstrainedOut: bool,
    currentConstrainedOut: Prefix,
    cost: int
  )
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires !insideConstrained ==> currentConstrained == []
    requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
    requires insideConstrained ==> |currentConstrained| <= |generatedPrefix|
    requires insideConstrained ==> generatedPrefix[|generatedPrefix| - |currentConstrained|..] == currentConstrained
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    ensures cost <= maxSteps
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained

  {
    var helpers := new CSDHelpers();
    generated := generatedPrefix;
    insideConstrainedOut := insideConstrained;
    currentConstrainedOut := currentConstrained;
    cost := 0;

    if maxSteps == 0 {
    } else if !insideConstrained {
      var continuation := helpers.CraneGeneration(lm, parser, prompt + generatedPrefix, maxSteps, 0, eosToken);
      generated := generatedPrefix + continuation;
      insideConstrainedOut := false;
      currentConstrainedOut := [];
      cost := 0;
    } else {
      var steps := 0;
      var localCurrent := currentConstrained;
      var generatedSuffix: Prefix := [];
      insideConstrainedOut := true;

      while steps < maxSteps && !parser.IsCompletePrefix(localCurrent)
        invariant 0 <= steps <= maxSteps
        invariant lm.ValidTokensIdsLogits()
        invariant parser.IsValidPrefix(localCurrent)
        invariant generated == generatedPrefix + generatedSuffix
        invariant |generatedSuffix| == steps
        invariant |localCurrent| <= |generated|
        invariant generated[|generated| - |localCurrent|..] == localCurrent
        invariant cost == 0
        invariant helpers.cost == steps
        invariant insideConstrainedOut
        decreases maxSteps - steps
      {
        var constrainedPrompt := prompt + generatedPrefix[..|generatedPrefix| - |currentConstrained|];
        var next, wasConstrained := helpers.ConfidenceGatedStep(
          lm, parser, constrainedPrompt, localCurrent, eosToken
        );
        if next == eosToken {
          break;
        }
        generated := generated + [next];
        generatedSuffix := generatedSuffix + [next];
        steps := steps + 1;
        if Contains(next, ">>") {
          insideConstrainedOut := false;
          localCurrent := [];
          break;
        } else {
          localCurrent := localCurrent + [next];
        }
      }

      currentConstrainedOut := localCurrent;
      if !insideConstrainedOut {
        currentConstrainedOut := [];
      }
      cost := 0;
    }

    if maxSteps > 0 {
      cost := 1;
    }
  }
}
