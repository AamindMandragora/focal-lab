include "../library/VerifiedAgentSynthesis.dfy"

// Reference reconstruction: greedy constrained decoding (GCD / SynCode-style).
// Every model-chosen token is grammar-constrained via hard masking.  There is no
// unconstrained reasoning: the strategy immediately opens a constrained span and
// stays there until the parse is complete.  Delimiters (<< / >>) are emitted by
// helpers (not by the model), keeping extraction uniform across benchmarks.
module ReferenceGcdCSD {
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

    // If not already inside a constrained span, open one immediately.
    // OpenConstrainedSpan emits << and enters constrained mode.
    if !inside {
      assert helpers.lm.Logits == old(lm.Logits);
      g, inside, cur := helpers.OpenConstrainedSpan(g);
      assert helpers.lm.Logits == old(lm.Logits);
      assert parser.IsValidPrefix(cur);
    }

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
      decreases maxSteps - helpers.cost()
    {
      if parser.IsCompletePrefix(cur) {
        // Parse is complete — close the span (emits >>) and stop.
        assert helpers.lm.Logits == old(lm.Logits);
        g, inside, cur := helpers.CloseConstrainedSpan(g, cur);
        assert helpers.lm.Logits == old(lm.Logits);
        break;
      }

      assert helpers.lm.Logits == old(lm.Logits);
      var next := helpers.ConstrainedStep(prompt, cur, eosToken);
      assert helpers.lm.Logits == old(lm.Logits);
      if next == eosToken {
        break;
      }
      assert helpers.lm.Logits == old(lm.Logits);
      g, inside, cur := helpers.AppendConstrainedToken(g, cur, next);
      assert helpers.lm.Logits == old(lm.Logits);
    }

    generated := g;
    insideConstrainedOut := inside;
    currentConstrainedOut := if inside then cur else [];
    cost := helpers.cost();
  }
}
