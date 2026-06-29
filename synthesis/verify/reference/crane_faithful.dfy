include "../library/VerifiedAgentSynthesis.dfy"

// Faithful CRANE baseline: delegates to the verified library CraneGeneration
// method (confidence-gated adaptive switching, substring-based << / >> span
// detection via Contains). This is the canonical CRANE baseline defined in
// CLAUDE.md, distinct from crane.dfy (which uses exact-equality `next == "<<"`
// span entry that misses the leading-space token on small tokenizers).
module ReferenceCraneFaithfulCSD {
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
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    ensures cost <= maxSteps
  {
    var helpers := new CSDHelpers();
    generated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, 10, eosToken);
    insideConstrainedOut := false;
    currentConstrainedOut := [];
    cost := helpers.cost;
  }
}
