include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, cost: int)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures cost <= maxSteps

  {
    var helpers := new CSDHelpers();
    // CSD_RATIONALE_BEGIN
// This strategy follows a two-phase approach for first-order logic reasoning problems. 
// In the unconstrained phase, it generates tokens freely until it encounters the "<<"
// token, indicating the start of a logical premise or conclusion. Upon encountering "<<",
// it transitions to the constrained phase, focusing on grammar-valid tokens until the
// ">>" token is encountered, indicating the end of a logical statement. If the end of the
// input is reached without completing the logic, it concludes with "Unknown".
// CSD_RATIONALE_END

generated := [];
var steps: nat := 0;
var insideConstrained: bool := false;
var currentConstrained: Prefix := [];

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant |generated| == steps
  invariant helpers.cost == steps
  invariant lm.ValidTokensIdsLogits()
  invariant insideConstrained ==> parser.IsValidPrefix(currentConstrained)
  invariant insideConstrained ==> forall t: Token :: t in parser.ValidNextTokens(currentConstrained) ==> t in lm.Tokens
  decreases maxSteps - steps, if insideConstrained then 1 else 0
{
  if !insideConstrained {
    // UNCONSTRAINED PHASE: choose HOW to generate the next unconstrained token
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    if next == eosToken { break; }
    generated := generated + [next];
    steps := steps + 1;
    if next == "<<" {
      insideConstrained := true;
      currentConstrained := [];
      CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, []);
    }
  } else {
    if parser.IsCompletePrefix(currentConstrained) {
      insideConstrained := false;
    } else {
      // CONSTRAINED PHASE: choose HOW to generate the next constrained token
      var next := helpers.ConstrainedStep(lm, parser, prompt, currentConstrained);
      generated := generated + [next];
      currentConstrained := currentConstrained + [next];
      steps := steps + 1;
    }
  }
}
cost := helpers.cost;
    cost := helpers.cost;
  }
}
