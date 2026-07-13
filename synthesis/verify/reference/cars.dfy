include "../library/VerifiedAgentSynthesis.dfy"

// Reference reconstruction: constrained adaptive rejection sampling (CARS).
//
// Models the full CARS algorithm within a single strategy invocation:
//   1.  constrain_first — the first constrained token is hard-masked.
//   2.  Exploration — subsequent tokens are sampled unconstrained
//       (SoftConstrainedStep with zero boost, matching CARS first-pass
//       behaviour on new trie nodes where grammar bitmask is stored but
//       not applied).
//   3.  Rejection — if the unconstrained token violates the grammar the
//       attempt is rejected, the failing token is recorded, and the
//       constrained suffix is rolled back to the span entry point
//       (models CARS raising ValueError and restarting from the trie root).
//   4.  Exploitation — on retries the accumulated rejected tokens are
//       penalised and generation is hard-masked (SafePenalizedConstrainedStep),
//       modelling CARS revisited trie nodes where log_theta carries both
//       the grammar bitmask and accumulated failure penalties.
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
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained

  {
    var helpers := new CSDHelpers();
    var g := generatedPrefix;
    var inside := insideConstrained;
    var cur := currentConstrained;

    var rejectedTokens: seq<Token> := [];
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
      invariant inside ==> spanEntryLen <= |generatedPrefix| + helpers.cost
      decreases maxSteps - helpers.cost
    {
      if inside && parser.IsCompletePrefix(cur) {
        g, inside, cur := helpers.CloseConstrainedSpan(lm, parser, g, cur);
      } else if !inside {
        var next := helpers.UnconstrainedStep(lm, prompt, g);
        g := g + [next];
        if next == eosToken {
          break;
        } else if next == "<<" {
          inside := true;
          cur := [];
          spanEntryLen := |g|;
          rejectedTokens := [];
        }
      } else if |cur| == 0 {
        // ---- constrain_first ------------------------------------------------
        // CARS always grammar-masks the first constrained token (root trie node
        // with constrain_first=True on first visit, grammar bitmask in log_theta
        // on revisits).
        var next := helpers.ConstrainedStep(lm, parser, prompt, cur, eosToken);
        if next == eosToken {
          break;
        }
        g, inside, cur := helpers.AppendConstrainedToken(lm, parser, g, cur, next);
      } else if |rejectedTokens| == 0 {
        // ---- exploration (first pass) ----------------------------------------
        // On new trie nodes at learn_level >= 3, CARS stores the grammar bitmask
        // in log_theta but does NOT apply it to scores (adjust_scores is False
        // for non-root new nodes).  Generation is effectively unconstrained.
        // Model this with SoftConstrainedStep(boost=0): sample from the raw LM
        // distribution, then check grammar validity.
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
          // ---- rejection (trie learning) ------------------------------------
          // CARS sets log_theta[failing_token] = -inf at the current trie node
          // and back-propagates through _recompute_in_trie, then raises
          // ValueError to abort the sample.  Model the token-level penalty as
          // an addition to rejectedTokens and the sample abort as a rollback
          // to the span entry point.
          lm.PenalizeTriedTokenAt(prompt + cur, next);
          rejectedTokens := rejectedTokens + [next];
          g := g[..spanEntryLen];
          cur := [];
        }
      } else {
        // ---- exploitation (revisited trie nodes) ----------------------------
        // On revisited nodes adjust_scores is True and scores += log_theta,
        // which contains both the grammar bitmask (-inf for invalid tokens)
        // and accumulated failure penalties.  SafePenalizedConstrainedStep
        // replicates this: hard grammar mask + penalty on rejected tokens.
        var next := helpers.SafePenalizedConstrainedStep(
          lm, parser, prompt, cur, rejectedTokens, 100000000.0, eosToken
        );
        if next == eosToken {
          break;
        }
        g, inside, cur := helpers.AppendConstrainedToken(lm, parser, g, cur, next);
      }
    }

    generated := g;
    insideConstrainedOut := inside;
    currentConstrainedOut := if inside then cur else [];
    cost := helpers.cost;
  }
}
