// Friction probe for the rollback helpers that return (generatedOut, currentOut)
// with an EQUATIONAL generatedOut postcondition but NO explicit
// `|currentOut| <= |generatedOut|` (unlike their siblings RollbackConstrainedSuffix
// and RollbackConstrainedToComplete, which state it). Question: can a caller already
// derive the length relation from the equational postcondition, or is there real
// friction? If both asserts pass on the baseline library -> no friction (no fix).
// If either fails -> real friction -> add the explicit ensures.
include "VerifiedAgentSynthesis.dfy"

module TestRollbackLengthRelation {
  import opened VerifiedDecoderAgent

  method ProbeRollbackConstrainedSpan(
    parser: Parser, stablePrefix: Prefix, generated: Prefix, currentConstrained: Prefix
  )
    requires parser.IsValidPrefix([])
    requires generated == stablePrefix + currentConstrained
  {
    var helpers := new CSDHelpers();
    var g, cur := helpers.RollbackConstrainedSpan(parser, stablePrefix, generated, currentConstrained);
    assert |cur| <= |g|;
  }

  method ProbeRollbackAndContinue(
    lm: LM, parser: Parser, prompt: Prefix, generated: Prefix,
    currentConstrained: Prefix, eosToken: Token, maxSteps: nat, closeReserve: nat, maxRetries: nat
  )
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires eosToken in lm.Tokens
    requires |currentConstrained| <= |generated|
    requires closeReserve <= maxSteps
  {
    var helpers := new CSDHelpers();
    var g, cur := helpers.RollbackAndContinue(
      lm, parser, prompt, generated, currentConstrained, eosToken, maxSteps, closeReserve, maxRetries);
    assert |cur| <= |g|;
  }
}
