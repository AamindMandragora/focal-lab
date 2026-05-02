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
    // CSD_RATIONALE_BEGIN
// The previous strategy was syntactically safe but too locally myopic: it took
// one constrained token at a time and relied mostly on plain sampling. On
// Spider, that produced two evaluation pathologies visible in the traces.
//
// First, several failures hit maxSteps with prefixes like `... >`, `... >=`,
// or other incomplete SQL tails. That means the decoder kept extending token by
// token without enough pressure to finish a clause and emit EOS/closure once a
// complete query became available.
//
// Second, the wrong answers were often semantically close but column/order
// choices were swapped. Always sampling from the full valid set leaves too much
// variance at exactly the schema-choice positions that matter for execution
// accuracy.
//
// This revision therefore switches to a more completion-oriented constrained
// policy:
//
// 1. Open the constrained SQL span immediately and stay inside it. Spider wants
//    exactly one SQL query, so there is no benefit to unconstrained narration.
//
// 2. Use AdaptiveConstrainedStep as the default action. It is still fully
//    grammar-masked, but when the parser-valid frontier is narrow it boosts the
//    caller-supplied token groups, which are most useful precisely at schema
//    selection points. When the frontier is broad, boosting is skipped, avoiding
//    the runaway identifier repetition seen in earlier attempts.
//
// 3. Add explicit narrow-frontier completion pressure. When the parser-valid
//    next-token count is small, we generate logits for the stable prompt plus
//    the current constrained prefix, boost valid groups moderately, strongly
//    penalize all schema/group tokens, then constrained-sample once. This keeps
//    operators/literals/closing structure competitive near the end of clauses,
//    reducing stalls like `WHERE cylinders >` and `HAVING avg(...) >=`.
//
// 4. Close immediately once the parser reports a complete query. EOS is also
//    terminal. This favors short complete SQL programs over endlessly extending
//    already-adequate prefixes.
//
// The key change from the failed attempt is replacing plain per-token
// ConstrainedStep with adaptive/group-aware decoding by default, plus a special
// low-branching completion branch that discourages another schema token when the
// query should likely terminate a predicate or subexpression.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: The open-span branch sets currentConstrainedOut := [] via
//   OpenConstrainedSpan, and parser.IsValidPrefix([]) holds. The close branch
//   sets insideConstrainedOut := false, so the implication is vacuous. In the
//   adaptive branch, AdaptiveConstrainedStep returns EOS or a parser-valid next
//   token; on non-EOS, AppendConstrainedToken extends to a valid prefix. In the
//   narrow completion branch, after GenerateLogits/boost/penalize,
//   ConstrainedSample guarantees EOS or a parser-valid next token; on non-EOS,
//   AppendConstrainedToken preserves validity.
// suffix: After opening, currentConstrainedOut is empty, so the suffix relation
//   holds trivially. After closing, insideConstrainedOut is false, so the
//   implication is vacuous. In both non-EOS token-producing branches,
//   AppendConstrainedToken appends the same token to generated and
//   currentConstrainedOut, preserving
//   generated[|generated|-|currentConstrainedOut|..] == currentConstrainedOut.
//   On EOS-breaking branches, state is unchanged.
// cost accounting: We maintain the invariant cost == 0 inside the loop and set
//   cost := steps at the end. OpenConstrainedSpan, AdaptiveConstrainedStep, and
//   ConstrainedSample each bump helper cost by 1; GenerateLogits, ValidTokenCount,
//   FlattenTokenGroups, IntersectTokenSets, BoostValidGroups, PenalizeTokenLogits,
//   and AppendConstrainedToken are non-bumping. Therefore every non-breaking
//   branch increments steps by exactly 1.
// progress bound: Opening appends exactly 1 token and increments steps by 1.
//   Closing appends at most 1 token, increments steps by 1, and breaks.
//   AdaptiveConstrainedStep either returns EOS and we break immediately, or we
//   append exactly 1 token with AppendConstrainedToken after increasing steps by
//   1. The narrow completion branch similarly either breaks on EOS or appends
//   exactly 1 token after steps := steps + 1. Hence every non-breaking branch
//   strictly increases steps and preserves |generated| <= |generatedPrefix| + steps.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant insideConstrainedOut ==> generated[|generated| - |currentConstrainedOut|..] == currentConstrainedOut
  invariant |generated| <= |generatedPrefix| + steps
  invariant cost == 0
  decreases maxSteps - steps
{
  if !insideConstrainedOut {
    var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);
    generated := openedGenerated;
    insideConstrainedOut := openedInside;
    currentConstrainedOut := openedCurrent;
    steps := steps + 1;
  } else {
    if parser.IsCompletePrefix(currentConstrainedOut) {
      var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := closedGenerated;
      insideConstrainedOut := closedInside;
      currentConstrainedOut := closedCurrent;
      steps := steps + 1;
      break;
    } else {
      var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
      var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
      if validCount <= 3 {
        var flatGroups := helpers.FlattenTokenGroups(validTokenGroups);
        var schemaTokens := helpers.IntersectTokenSets(flatGroups, lm.Tokens);
        lm.GenerateLogits(prompt + stablePrefix + currentConstrainedOut);
        helpers.BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, 2.0);
        helpers.PenalizeTokenLogits(lm, schemaTokens, 8.0);
        var next := helpers.ConstrainedSample(lm, parser, currentConstrainedOut, eosToken);
        steps := steps + 1;
        if next == eosToken {
          break;
        } else {
          var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
            lm, parser, generated, currentConstrainedOut, next
          );
          generated := appendedGenerated;
          insideConstrainedOut := appendedInside;
          currentConstrainedOut := appendedCurrent;
        }
      } else {
        var next := helpers.AdaptiveConstrainedStep(
          lm, parser, prompt + stablePrefix, currentConstrainedOut, validTokenGroups, 4.0, 10, eosToken
        );
        steps := steps + 1;
        if next == eosToken {
          break;
        } else {
          var appendedGenerated2, appendedInside2, appendedCurrent2 := helpers.AppendConstrainedToken(
            lm, parser, generated, currentConstrainedOut, next
          );
          generated := appendedGenerated2;
          insideConstrainedOut := appendedInside2;
          currentConstrainedOut := appendedCurrent2;
        }
      }
    }
  }
}

cost := steps;
  }
}
