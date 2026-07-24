include "../library/VerifiedAgentSynthesis.dfy"

module ReferenceCarsCSD {
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
  {
    var helpers := new CSDHelpers();
    var g := generatedPrefix;
    var inside := insideConstrained;
    var cur := currentConstrained;
    var spanEntryLen := if inside then |g| - |cur| else 0;

    if maxSteps == 0 {
      generated := g;
      insideConstrainedOut := inside;
      currentConstrainedOut := if inside then cur else [];
      cost := helpers.cost;
      return;
    }

    while helpers.cost < maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant |g| <= |generatedPrefix| + helpers.cost
      invariant !inside ==> cur == []
      invariant inside ==> parser.IsValidPrefix(cur)
      invariant inside ==> |cur| <= |g|
      invariant inside ==> g[|g| - |cur|..] == cur
      invariant 0 <= helpers.cost <= maxSteps
      invariant inside ==> 0 <= spanEntryLen <= |g|
      decreases maxSteps - helpers.cost
    {
      if inside && parser.IsCompletePrefix(cur) && parser.ValidNextTokenCount(cur) == 0 {
        // Token-0 / CARS raw SMILES: do not emit visible ">>".
        // Only stop when accept state has no further grammar continuations
        // (bare "C" can be complete yet still extendable).
        break;
      } else if !inside {
        var next := helpers.UnconstrainedStep(lm, prompt, g);
        g := g + [next];
        if next == eosToken { break; }
        else if next == "<<" { inside := true; cur := []; spanEntryLen := |g|; }
      } else {
        var constrainFirst := |cur| == 0;
        var next: Token;
        var ok: bool;
        next, ok := helpers.CarsTrieStep(lm, parser, prompt, cur, eosToken, constrainFirst);
        if !ok {
          // CARS get_sample: fail this full attempt; caller retries with updated trie.
          helpers.RejectLastInTrieHelper(lm);
          g := g[..spanEntryLen];
          cur := [];
          break;
        } else if next == eosToken {
          break;
        } else if parser.IsValidPrefix(cur + [next]) {
          g, inside, cur := helpers.AppendConstrainedToken(lm, parser, g, cur, next);
        } else {
          helpers.RejectLastInTrieHelper(lm);
          g := g[..spanEntryLen];
          cur := [];
          break;
        }
      }
    }

    generated := g;
    insideConstrainedOut := inside;
    currentConstrainedOut := if inside then cur else [];
    cost := helpers.cost;
  }
}
