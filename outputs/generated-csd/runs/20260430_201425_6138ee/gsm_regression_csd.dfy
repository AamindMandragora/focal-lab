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
// Math-step CSD with delimiter-triggered arithmetic spans. The strategy tracks
// the usual generation state (`generated`, `insideConstrainedOut`,
// `currentConstrainedOut`) and additionally uses the caller-supplied
// `validTokenGroups` only inside constrained spans to softly bias decoding
// toward arithmetic/operator/number-like token groups when the grammar is
// narrow. Outside spans it generates freely until either EOS or the token "<<"
// appears, which opens a constrained arithmetic computation region.
// 
// The main decision rule is:
//   * outside a span: unconstrained token generation;
//   * inside a span and the parser says the span is complete: close with ">>";
//   * otherwise inside a span: use an adaptive constrained step that boosts
//     caller-provided groups when the valid-next set is narrow.
// 
// This supports the task because each arithmetic computation is intended to be
// written inside << >>, and the parser-valid constrained prefix ensures the
// contents of an active span remain syntactically acceptable while still
// letting the LM choose among valid continuations.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained branch, the only transition into a
//   constrained span is when next == "<<"; then we set currentConstrainedOut :=
//   [], which is valid by the method precondition parser.IsValidPrefix([]). In
//   the close branch, CloseConstrainedSpan sets insideConstrainedOut to false,
//   so the implication becomes vacuous. In the adaptive constrained branch,
//   AdaptiveConstrainedStep guarantees either EOS or a parser-valid next token;
//   on EOS we break with state unchanged, and on non-EOS AppendConstrainedToken
//   yields a parser-valid updated constrained prefix.
// suffix: In the unconstrained branch, if we remain outside the span the
//   implication is vacuous; if we open a span on "<<", currentConstrainedOut is
//   [] so the length-0 suffix of generated matches []. In the close branch,
//   CloseConstrainedSpan appends at most one delimiter token and resets the
//   constrained prefix to [], making the suffix implication vacuous. In the
//   adaptive constrained branch, AppendConstrainedToken appends the same token
//   to both generated and currentConstrainedOut, preserving the suffix
//   equality; the EOS sub-branch breaks before changing state.
// cost accounting: We return cost := steps at the end, so it suffices that
//   every non-breaking iteration increments steps exactly by the amount of work
//   performed. UnconstrainedStep, CloseConstrainedSpan, and
//   AdaptiveConstrainedStep are all cost-bumping helpers with unit cost, and
//   each corresponding non-breaking branch does steps := steps + 1. The
//   AppendConstrainedToken helper is non-bumping, so the constrained branch's
//   accounting is still exactly the one unit already charged by
//   AdaptiveConstrainedStep.
// progress bound: In the unconstrained branch, generated grows by exactly one
//   token when we do not break, and steps increases by 1. In the close branch,
//   CloseConstrainedSpan appends at most one token and steps increases by 1. In
//   the adaptive constrained branch, EOS breaks immediately; otherwise
//   AppendConstrainedToken adds exactly one token and steps increases by 1, so
//   |generated| <= |generatedPrefix| + steps is preserved and the decreases
//   metric maxSteps - steps strictly shrinks on every non-breaking path.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var narrowThreshold: nat := 12;

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
      var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.AdaptiveConstrainedStep(
        lm, parser, prompt + stablePrefix, currentConstrainedOut, validTokenGroups, 4.0, narrowThreshold, eosToken
      );
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

cost := steps;
  }
}
