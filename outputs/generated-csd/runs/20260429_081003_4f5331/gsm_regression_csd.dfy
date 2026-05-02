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
    generated := generatedPrefix;
    insideConstrainedOut := insideConstrained;
    currentConstrainedOut := currentConstrained;
    cost := 0;
    // CSD_RATIONALE_BEGIN
// Math-step CSD with delimiter-aware outside generation and adaptive inside
// decoding. The strategy tracks the usual constrained-span state
// (insideConstrainedOut, currentConstrainedOut) plus a lightweight local signal
// about whether the current constrained arithmetic span looks narrow according
// to the parser's valid-next count.
//
// Outside a span, generation is unconstrained except that the model may emit
// the delimiter token "<<" to begin an arithmetic computation. Inside a span,
// if the parser says the current arithmetic fragment is already complete, we
// immediately close it with ">>". Otherwise we choose between two modes:
// token-level constrained decoding when the parser-valid continuation set is
// small, or a bounded ConstrainedSymbol expansion when the grammar is wider.
// This encourages natural multi-token arithmetic fragments while still keeping
// every token inside << >> parser-valid.
//
// Minimal verification fix: the only problematic branch was the wide inside
// branch, where steps could increase by stepsUsed without any proof that
// stepsUsed <= maxSteps - steps. We keep the same strategy but cap the symbol
// budget by the remaining step budget, so the returned stepsUsed is provably
// within the loop bound. This restores the loop invariant steps <= maxSteps and
// therefore the generated-length and returned-cost postconditions.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the outside-span branch, we only enter constrained mode
//   when next == "<<", and then set currentConstrainedOut := [], which is a
//   valid prefix by precondition. In the complete-prefix branch,
//   CloseConstrainedSpan sets insideConstrainedOut to false, so the invariant
//   becomes vacuous. In the narrow inside branch, ConstrainedStep returns a
//   parser-valid next token (or EOS, which breaks), and AppendConstrainedToken
//   preserves validity. In the wide inside branch, ConstrainedSymbol returns
//   currentOut that is itself a valid parser prefix, and we assign that to
//   currentConstrainedOut.
// suffix: Outside the span, either we stay outside (implication vacuous) or we
//   open on "<<", set currentConstrainedOut := [], and the length-0 suffix of
//   generated matches []. CloseConstrainedSpan appends the closing delimiter
//   and resets currentConstrainedOut to [], so the implication is again
//   vacuous. In the narrow inside branch, AppendConstrainedToken appends the
//   same token to generated and currentConstrainedOut. In the wide inside
//   branch, we compute stablePrefix := generated[..|generated|-|current|] and
//   then set generated := stablePrefix + symbolOut and currentConstrainedOut :=
//   symbolOut, so generated's suffix is exactly currentConstrainedOut.
// cost accounting: We return cost := steps, with steps incremented exactly by
//   the amount of decoding work performed in each non-break branch: +1 for
//   UnconstrainedStep, CloseConstrainedSpan, and ConstrainedStep; +stepsUsed
//   for ConstrainedSymbol. Pure queries do not affect steps. The only extra
//   proof obligation is steps <= maxSteps, handled by capping the wide-branch
//   symbol budget to the remaining budget before calling ConstrainedSymbol.
// progress: Outside, narrow-inside, and close-span branches append at most one
//   token and increment steps by 1, so |generated| <= |generatedPrefix| +
//   steps is preserved by linear arithmetic. In the wide inside branch,
//   ConstrainedSymbol extends the constrained fragment by at most stepsUsed
//   tokens, and we increase steps by exactly stepsUsed; rebuilding generated as
//   stablePrefix + symbolOut therefore preserves the same bound. Break branches
//   leave generated unchanged.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var narrowThreshold: int := 8;

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
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      generated := generated + [next];
      if next == "<<" {
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }
    }
  } else {
    if parser.IsCompletePrefix(currentConstrainedOut) {
      var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
        lm, parser, generated, currentConstrainedOut
      );
      generated := closedGenerated;
      insideConstrainedOut := closedInside;
      currentConstrainedOut := closedCurrent;
      steps := steps + 1;
    } else {
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
      if validCount <= narrowThreshold || stepTokenBudget == 0 || maxSteps - steps == 1 {
        var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
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
        var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
        var remaining: nat := maxSteps - steps;
        var symbolBudget: nat := if stepTokenBudget < remaining then stepTokenBudget else remaining;
        var symbolOut, hitEos, stepsUsed := helpers.ConstrainedSymbol(
          lm, parser, constrainedPrompt, currentConstrainedOut, symbolBudget, eosToken
        );
        generated := stablePrefix + symbolOut;
        insideConstrainedOut := true;
        currentConstrainedOut := symbolOut;
        steps := steps + stepsUsed;
        if hitEos {
          break;
        }
      }
    }
  }
}

cost := steps;
    if maxSteps > 0 && cost == 0 { cost := 1; }  // guarantee progress postcondition
  }
}
