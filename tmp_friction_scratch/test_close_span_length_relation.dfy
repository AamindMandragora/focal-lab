// TDD vehicle for FIX 1 of the helper verification-friction audit (2026-06-18).
//
// Reproduces the EXACT proof obligation that the groundbudget att3 strategy could
// not discharge: after calling CloseSpanWithinBudget inside a loop, the caller must
// re-establish its loop invariant `insideConstrainedOut ==> |currentConstrainedOut|
// <= |generated|` (mirrors the template precondition GeneratedCSD.dfy:28). The helper's
// current contract never relates |currentOut| to |generatedOut|, so the caller cannot
// prove it -> verify fail (`GeneratedCSD.dfy(160,61): this invariant could not be proved
// to be maintained by the loop`).
//
// The assert below IS that missing fact. RED on the baseline library; GREEN once
// CloseSpanWithinBudget gains `ensures insideOut ==> |currentOut| <= |generatedOut|`.
include "VerifiedAgentSynthesis.dfy"

module TestCloseSpanLengthRelation {
  import opened VerifiedDecoderAgent

  method ProbeCloseSpanLengthRelation(
    lm: LM, parser: Parser, prompt: Prefix, generated: Prefix,
    currentConstrained: Prefix, eosToken: Token, budget: nat
  )
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix(currentConstrained)
    requires |currentConstrained| <= |generated|
    requires eosToken in lm.Tokens
    requires ">>" in lm.Tokens
  {
    var helpers := new CSDHelpers();
    var g, ins, cur := helpers.CloseSpanWithinBudget(
      lm, parser, prompt, generated, currentConstrained, eosToken, budget);
    // The relational fact the caller's loop invariant needs and the helper must supply:
    assert ins ==> |cur| <= |g|;
  }
}
