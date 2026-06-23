include "../library/VerifiedAgentSynthesis.dfy"

// Reference reconstruction: RS (standard rejection sampling).
//
// Within a constrained span, each token is drawn from the raw LM distribution
// (SoftConstrainedStep with zero boost, i.e. no grammar mask during sampling).
// If the draw violates the grammar, the entire candidate prefix inside the span
// is discarded and generation restarts from the span entry point — there is no
// accumulated penalty state (unlike CARS exploitation with SafePenalizedConstrainedStep).
module ReferenceRsCSD {
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
    lm.SetUseSampling(true);
    assert lm.Logits == old(lm.Logits);
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

    if !inside {
      assert helpers.lm.Logits == old(lm.Logits);
      g, inside, cur := helpers.OpenConstrainedSpan(g);
      assert helpers.lm.Logits == old(lm.Logits);
      assert parser.IsValidPrefix(cur);
    }

    var spanEntryLen := |g| - |cur|;

    while helpers.cost() < maxSteps
      modifies lm, old(lm.Logits)
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant lm.Logits == old(lm.Logits)
      invariant fresh(helpers)
      invariant |g| <= |generatedPrefix| + helpers.cost()
      invariant inside
      invariant parser.IsValidPrefix(cur)
      invariant |cur| <= |g|
      invariant inside ==> g[|g| - |cur|..] == cur
      invariant 0 <= helpers.cost() <= maxSteps
      invariant 0 <= spanEntryLen <= |g|
      decreases maxSteps - helpers.cost()
    {
      if parser.IsCompletePrefix(cur) {
        assert helpers.lm.Logits == old(lm.Logits);
        g, inside, cur := helpers.CloseConstrainedSpan(g, cur);
        assert helpers.lm.Logits == old(lm.Logits);
        break;
      }

      var next: Token;
      var isValid: bool;
      assert helpers.lm.Logits == old(lm.Logits);
      next, isValid := helpers.SoftConstrainedStep(prompt, cur, 0.0, eosToken
      );
      assert helpers.lm.Logits == old(lm.Logits);
      if next == eosToken {
        break;
      }
      if isValid {
        assert helpers.lm.Logits == old(lm.Logits);
        g, inside, cur := helpers.AppendConstrainedToken(g, cur, next);
        assert helpers.lm.Logits == old(lm.Logits);
      } else {
        g := g[..spanEntryLen];
        cur := [];
      }
    }

    generated := g;
    insideConstrainedOut := inside;
    currentConstrainedOut := if inside then cur else [];
    cost := helpers.cost();
  }
}
