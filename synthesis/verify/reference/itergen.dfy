include "../library/VerifiedAgentSynthesis.dfy"

// Reference reconstruction: speculative unconstrained step with grammar-mask
// fallback at every constrained token (IterGen-style).
module ReferenceIterGenCSD {
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
    var helpers := new CSDHelpers(lm, parser);
    assert helpers.lm.Logits == old(lm.Logits);
    assert helpers.lm == lm;
    assert helpers.parser == parser;
    assert lm.ValidTokensIdsLogits();
    var g := generatedPrefix;
    var inside := insideConstrained;
    var cur := currentConstrained;

    if maxSteps == 0 {
      generated := g;
      insideConstrainedOut := inside;
      currentConstrainedOut := if inside then cur else [];
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
      invariant !inside ==> cur == []
      invariant inside ==> parser.IsValidPrefix(cur)
      invariant inside ==> |cur| <= |g|
      invariant inside ==> g[|g| - |cur|..] == cur
      invariant 0 <= helpers.cost() <= maxSteps
      decreases maxSteps - helpers.cost()
    {
      if inside && parser.IsCompletePrefix(cur) {
        assert helpers.lm.Logits == old(lm.Logits);
        g, inside, cur := helpers.CloseConstrainedSpan(g, cur);
        assert helpers.lm.Logits == old(lm.Logits);
      } else if !inside {
        assert helpers.lm.Logits == old(lm.Logits);
        var next := helpers.UnconstrainedStep(prompt, g);
        assert helpers.lm.Logits == old(lm.Logits);
        g := g + [next];
        if next == eosToken {
          break;
        } else if next == "<<" {
          inside := true;
          cur := [];
        }
      } else {
        var next: Token;
        var fb: bool;
        assert helpers.lm.Logits == old(lm.Logits);
        next, fb := helpers.SafeSoftConstrainedStep(prompt, cur, 0.0, eosToken
        );
        assert helpers.lm.Logits == old(lm.Logits);
        if next == eosToken {
          break;
        }
        assert helpers.lm.Logits == old(lm.Logits);
        g, inside, cur := helpers.AppendConstrainedToken(g, cur, next);
        assert helpers.lm.Logits == old(lm.Logits);
      }
    }

    generated := g;
    insideConstrainedOut := inside;
    currentConstrainedOut := if inside then cur else [];
    cost := helpers.cost();
  }
}
