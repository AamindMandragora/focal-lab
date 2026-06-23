include "../library/VerifiedAgentSynthesis.dfy"

// Reference reconstruction: pure unconstrained decoding (no grammar enforcement).
module ReferenceUnconstrainedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(
    lm: LM,
    parser: Parser,
    prompt: Prefix,
    generatedPrefix: Prefix,
    insideConstrained: bool,
    currentConstrained: Prefix,
    maxSteps: nat,
    stepTokenBudget: nat,
    validTokenGroups: seq<seq<Token>>,
    eosToken: Token
  ) returns (
    generated: Prefix,
    insideConstrainedOut: bool,
    currentConstrainedOut: Prefix,
    cost: int
  )
    modifies lm, lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires !insideConstrained ==> currentConstrained == []
    requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
    requires insideConstrained ==> |currentConstrained| <= |generatedPrefix|
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
    var helpers := new CSDHelpers(lm, parser);
    assert helpers.lm.Logits == old(lm.Logits);
    assert helpers.lm == lm;
    assert helpers.parser == parser;
    assert lm.ValidTokensIdsLogits();
    var g := generatedPrefix;

    if maxSteps == 0 {
      generated := g;
      insideConstrainedOut := false;
      currentConstrainedOut := [];
      cost := helpers.cost();
      return;
    }

    while helpers.cost() < maxSteps
      modifies lm, old(lm.Logits)
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant lm.Logits == old(lm.Logits)
      invariant fresh(helpers) 
      invariant |g| <= |generatedPrefix| + helpers.cost()
      invariant 0 <= helpers.cost() <= maxSteps
      decreases maxSteps - helpers.cost()
    {
      assert helpers.lm.Logits == old(lm.Logits);
      var next := helpers.UnconstrainedStep(prompt, g);
      assert helpers.lm.Logits == old(lm.Logits);
      g := g + [next];
      if next == eosToken {
        break;
      }
    }

    generated := g;
    insideConstrainedOut := false;
    currentConstrainedOut := [];
    cost := helpers.cost();
  }
}
