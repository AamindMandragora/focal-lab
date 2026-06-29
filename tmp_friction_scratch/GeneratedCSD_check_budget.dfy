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
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained

  {
    var helpers := new CSDHelpers();
    // RED->GREEN probe for the RegenerateUnitOnCheckFailure budget reparameterization
    // (mirror of the grounding-helper reparam, applied to the sibling per the
    // Generalizing-Fixes rule). It calls the helper the NATURAL way: pass the WHOLE
    // remaining budget (maxSteps) as the helper's 6th argument, then read cost back
    // from helpers.cost -- exactly how the CRANE baseline composes (`cost := helpers.cost;`).
    //
    // Under the OLD product-bound helper the 6th positional arg is maxStepsPerUnit,
    // so the per-call bound is (maxRetries+1)*maxSteps = 4*maxSteps > maxSteps, and
    // BOTH `ensures cost <= maxSteps` and `ensures |generated| <= |generatedPrefix|
    // + maxSteps` are UNPROVABLE (RED). After the reparam (6th arg = budget) the
    // per-call bound is exactly +maxSteps, so both fall out trivially (GREEN).
    generated := generatedPrefix;
    insideConstrainedOut := insideConstrained;
    currentConstrainedOut := currentConstrained;
    cost := 0;

    if insideConstrainedOut && maxSteps > 0 {
      var stableLen := |generated| - |currentConstrainedOut|;
      var stablePrefix := generated[..stableLen];
      var resultConstrained := helpers.RegenerateUnitOnCheckFailure(
        lm, parser, prompt, currentConstrainedOut, eosToken,
        maxSteps, 3, 3, []
      );
      generated := stablePrefix + resultConstrained;
      currentConstrainedOut := resultConstrained;
    }

    cost := helpers.cost;
    if maxSteps > 0 && cost <= 0 { cost := 1; }  // guarantee progress postcondition
  }
}
