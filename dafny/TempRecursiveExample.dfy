include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, currentPrefix: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, cost: int)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix(currentPrefix)
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures parser.IsValidPrefix(generated)
    ensures exists suffix: Prefix :: generated == currentPrefix + suffix
    ensures |generated| <= |currentPrefix| + maxSteps
    ensures cost <= maxSteps
    decreases maxSteps
  {
    var helpers := new CSDHelpers();
    generated := currentPrefix;
    cost := 0;
    var suffix: Prefix := [];

    if maxSteps == 0 || parser.IsCompletePrefix(currentPrefix) {
      suffix := [];
      generated := currentPrefix;
      assert parser.IsValidPrefix(generated);
      cost := 0;
    } else {
      helpers.cost := 0;
      var candidates := helpers.TopValidCandidates(lm, parser, prompt, currentPrefix, 2, eosToken);
      var childBudget: nat := (maxSteps - 1) / 2;

      var first := candidates[0];
      var best := currentPrefix;
      var bestSuffix: Prefix := [];
      var bestCost: int := helpers.cost;

      if first != eosToken {
        assert first in candidates;
        assert first in parser.ValidNextTokens(currentPrefix);
        var firstPrefix := currentPrefix + [first];
        assert parser.IsValidPrefix(firstPrefix);
        var firstGenerated, firstCost := MyCSDStrategy(
          lm, parser, prompt, firstPrefix, childBudget, eosToken
        );
        var firstSuffix :| firstGenerated == firstPrefix + firstSuffix;
        best := firstGenerated;
        bestSuffix := [first] + firstSuffix;
        bestCost := helpers.cost + firstCost;
      }

      if |candidates| > 1 {
        var second := candidates[1];
        if second != eosToken {
          assert second in candidates;
          assert second in parser.ValidNextTokens(currentPrefix);
          var secondPrefix := currentPrefix + [second];
          assert parser.IsValidPrefix(secondPrefix);
          var secondGenerated, secondCost := MyCSDStrategy(
            lm, parser, prompt, secondPrefix, childBudget, eosToken
          );
          var secondSuffix :| secondGenerated == secondPrefix + secondSuffix;
          var secondTotal := helpers.cost + secondCost;
          if parser.IsCompletePrefix(secondGenerated) && !parser.IsCompletePrefix(best) {
            best := secondGenerated;
            bestSuffix := [second] + secondSuffix;
            bestCost := secondTotal;
          } else if parser.IsCompletePrefix(secondGenerated) == parser.IsCompletePrefix(best) &&
                    |secondGenerated| > |best| {
            best := secondGenerated;
            bestSuffix := [second] + secondSuffix;
            bestCost := secondTotal;
          } else {
            assert bestCost <= helpers.cost + childBudget;
          }
        }
      }

      assert 2 * childBudget <= maxSteps - 1;
      assert helpers.cost == 1;
      assert bestCost <= maxSteps;
      assert best == currentPrefix + bestSuffix;
      assert |best| <= |currentPrefix| + maxSteps;
      suffix := bestSuffix;
      generated := currentPrefix + suffix;
      assert generated == best;
      assert parser.IsValidPrefix(generated);
      cost := bestCost;
    }
    assert generated == currentPrefix + suffix;
  }
}
