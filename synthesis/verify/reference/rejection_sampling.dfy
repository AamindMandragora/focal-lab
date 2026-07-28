include "../library/VerifiedAgentSynthesis.dfy"

// Reference reconstruction: rejection sampling (RS).
//
// Models the harness RS baseline inside a single strategy invocation:
//   1.  Free prefix — tokens outside a constrained span are sampled without
//       any grammar control, exactly as in unconstrained decoding.
//   2.  Proposal — inside the span every token is drawn from the raw LM
//       distribution (SoftConstrainedStep with zero boost never shifts the
//       logits), then checked against the parser after the fact.
//   3.  Rejection — a proposal that leaves the constrained prefix invalid
//       discards the whole attempt: the constrained suffix is truncated back
//       to the span entry point and the next attempt starts from scratch.
//
// The contrast with cars.dfy is the absence of state: no failing token is
// recorded, no logit penalty is accumulated, and no token is ever hard-masked,
// so every retry samples from the same distribution as the first attempt.
// Retries are bounded only by the shared token-step budget maxSteps.
module ReferenceRejectionSamplingCSD {
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
      invariant inside ==> spanEntryLen == |g| - |cur|
      decreases maxSteps - helpers.cost
    {
      if inside && parser.IsCompletePrefix(cur) {
        // Accepted sample: close the span and stop rejecting.
        g, inside, cur := helpers.CloseConstrainedSpan(lm, parser, g, cur);
      } else if !inside {
        // ---- free prefix ----------------------------------------------------
        // No grammar control outside the span; entering a span records the
        // rollback point that a rejected attempt returns to.
        var next := helpers.UnconstrainedStep(lm, prompt, g);
        g := g + [next];
        if next == eosToken {
          break;
        } else if next == "<<" {
          inside := true;
          cur := [];
          spanEntryLen := |g|;
        }
      } else {
        // ---- proposal and post-hoc check -------------------------------------
        // SoftConstrainedStep with boostAmount = 0.0 adds nothing to the
        // logits, so the draw is distributed exactly as an unconstrained draw;
        // isValid reports whether the parser still accepts the extended prefix.
        var next: Token;
        var isValid: bool;
        next, isValid := helpers.SoftConstrainedStep(
          lm, parser, prompt, cur, 0.0, eosToken
        );
        if next == eosToken {
          break;
        }
        if isValid {
          g, inside, cur := helpers.AppendConstrainedToken(lm, parser, g, cur, next);
        } else {
          // ---- rejection (memoryless restart) --------------------------------
          // Discard the whole attempt and resume from the span entry point.
          // Nothing about the failure is remembered, so the next attempt draws
          // from the same distribution as this one.
          g := g[..spanEntryLen];
          cur := [];
        }
      }
    }

    generated := g;
    insideConstrainedOut := inside;
    currentConstrainedOut := if inside then cur else [];
    cost := helpers.cost;
  }
}
