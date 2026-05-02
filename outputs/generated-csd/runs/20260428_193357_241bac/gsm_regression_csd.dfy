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
// Step-by-step math CSD with delimiter-aware arithmetic spans. The strategy
// tracks the usual generated/current constrained suffix state, plus a local
// arithmetic-preference bag derived from validTokenGroups. Outside a span it
// generates freely until either EOS or the open delimiter "<<" appears.
// Inside a span it asks whether the parser prefix is complete; if so it closes
// the span immediately with ">>". Otherwise it performs constrained decoding.
//
// The constrained decoding mode is adaptive. When the parser's valid-next set
// is narrow, it uses ConstrainedStep directly. When the set is wider, it
// generates logits for the constrained prefix, optionally boosts caller-supplied
// preferred arithmetic tokens (digits/operators/etc.) from validTokenGroups,
// hard-masks to parser-valid next tokens plus EOS, and samples once. This keeps
// arithmetic text inside << >> parser-valid while still allowing task-specific
// token preferences supplied by the caller.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained branch, entering a span only happens
//   when next == "<<", and then currentConstrainedOut := [], which is valid by
//   precondition parser.IsValidPrefix([]). In the complete-prefix branch,
//   CloseConstrainedSpan sets insideConstrainedOut to false, so the implication
//   is vacuous. In the narrow constrained branch, ConstrainedStep returns a
//   parser-valid token (or EOS, which breaks), and AppendConstrainedToken
//   preserves validity. In the wide constrained branch, we call
//   MaskValidNextInGroup only as a logit preference, then MaskValidNextAndEos;
//   therefore any non-EOS token chosen by ChooseNextToken is parser-valid, and
//   AppendConstrainedToken preserves validity.
//
// suffix: In the unconstrained branch, when we stay outside the span the
//   implication is vacuous; when we enter on "<<", currentConstrainedOut is []
//   so the length-0 suffix condition holds immediately. In the complete-prefix
//   branch, CloseConstrainedSpan appends the closing delimiter and resets the
//   constrained suffix to [], so the implication is vacuous afterward. In both
//   constrained-token branches, AppendConstrainedToken appends the same token to
//   generated and currentConstrainedOut, preserving the suffix equality.
//
// progress: UnconstrainedStep, CloseConstrainedSpan, and the narrow
//   ConstrainedStep branch each consume one step and append at most one token,
//   so |generated| <= |generatedPrefix| + steps is preserved by linear
//   arithmetic. In the wide constrained branch, ChooseNextToken consumes one
//   step and either breaks on EOS or AppendConstrainedToken appends exactly one
//   token, so the same bound is preserved. Branches that break do not need to
//   re-establish future-loop invariants.
//
// cost accounting: The loop uses a local steps counter only. Every non-break
//   action branch increments steps by exactly 1 after the corresponding
//   cost-bumping helper or primitive sample. The returned cost is assigned as
//   cost := steps after the loop, so cost <= maxSteps follows from the loop
//   bound 0 <= steps <= maxSteps.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var wideThreshold: nat := 8;
var preferredFlat: seq<Token> := helpers.FlattenTokenGroups(validTokenGroups);

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
      if validCount <= wideThreshold {
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
        lm.GenerateLogits(constrainedPrompt + currentConstrainedOut);
        if |preferredFlat| > 0 {
          var anyValid := helpers.MaskValidNextInGroup(
            lm, parser, currentConstrainedOut, preferredFlat, eosToken
          );
        }
        lm.MaskValidNextAndEos(parser, currentConstrainedOut, eosToken);
        var next := lm.ChooseNextToken();
        helpers.cost := helpers.cost + 1;
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
      }
    }
  }
}

cost := steps;
    if maxSteps > 0 && cost == 0 { cost := 1; }  // guarantee progress postcondition
  }
}
