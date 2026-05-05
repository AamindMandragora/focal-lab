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
// Refinement mode: valid-basin single-axis repair. I preserve the evidence-backed
// ingredients that improved syntax and reduced stalls: bounded free chunking
// outside spans, observed span entry when "<<" is naturally emitted, symbol-level
// constrained progress in wider parser states, narrow hard-token fallback, and
// immediate close on parser-complete prefixes.
//
// Measured failure source to change: the previous strategy almost never produced
// a complete visible final span and often continued with free text after a valid
// constrained chunk. The causal axis changed here is span-entry policy outside
// constrained regions: instead of waiting almost entirely for a naturally emitted
// "<<", the strategy arms an explicit open after chemistry-answer cues in the
// surrounding text. This imports only the cue-triggered entry idea from the older
// family while preserving the balanced-best inside-span mechanics.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the outside-span explicit-open branch, OpenConstrainedSpan
//   sets currentConstrainedOut := [], which is a valid parser prefix. In the
//   outside-span chunk branch, insideConstrainedOut stays false unless the chunk
//   already emitted "<<"; then EnterObservedConstrainedSpan resets the active
//   constrained content to [], again a valid prefix. In the close branch,
//   CloseConstrainedSpan exits constrained mode, so the implication becomes
//   vacuous. In the narrow constrained branch, AdaptiveConstrainedStep returns
//   EOS or a parser-valid next token, and AppendConstrainedToken preserves
//   parser.IsValidPrefix. In the wider constrained branch,
//   ConstrainedSymbolInGenerated returns a parser-valid constrained prefix, so
//   assigning its currentOut preserves the invariant.
//
// progress: In the explicit-open branch, steps increases by 1 and
//   OpenConstrainedSpan appends exactly one visible delimiter, so
//   |generated| <= |generatedPrefix| + steps is preserved. In the outside-span
//   chunk branch, steps increases by stepsUsed and UnconstrainedChunk appends at
//   most that many visible tokens, so the bound is preserved. In the close
//   branch, CloseConstrainedSpan consumes one token-step and appends at most one
//   visible delimiter. In the narrow constrained branch,
//   AdaptiveConstrainedStep consumes one token-step; EOS appends nothing, and a
//   non-EOS token committed by AppendConstrainedToken adds one visible token. In
//   the wider constrained branch, ConstrainedSymbolInGenerated consumes
//   stepsUsed token budget, possibly including EOS or rejected suffix tokens,
//   while visible growth is at most that consumed budget, so the output-length
//   bound still holds.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var narrowThreshold: nat := 12;
var cueArmed: bool := false;

var lastTok, foundLast := helpers.LastTokenBefore(generated, ">>");
if foundLast {
  if lastTok == ":" || lastTok == "SMILES" || lastTok == "smiles" || lastTok == "=" || lastTok == "Answer" || lastTok == "answer" {
    cueArmed := true;
  }
}

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  if !insideConstrainedOut {
    if cueArmed {
      var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);
      generated := openedGenerated;
      insideConstrainedOut := openedInside;
      currentConstrainedOut := openedCurrent;
      cueArmed := false;
      steps := steps + 1;
    } else {
      var chunkBudget: nat := if maxSteps - steps > 4 then 4 else maxSteps - steps;
      var chunkedGenerated, stoppedOnOpenSpan, stoppedOnEos, stepsUsed := helpers.UnconstrainedChunk(
        lm, prompt, generated, chunkBudget, "<<", eosToken
      );
      generated := chunkedGenerated;
      steps := steps + stepsUsed;

      var newLastTok, foundNewLast := helpers.LastTokenBefore(generated, ">>");
      if foundNewLast {
        if newLastTok == ":" || newLastTok == "SMILES" || newLastTok == "smiles" || newLastTok == "=" || newLastTok == "Answer" || newLastTok == "answer" {
          cueArmed := true;
        }
      }

      if stoppedOnEos {
        break;
      } else if stoppedOnOpenSpan {
        var enteredGenerated, enteredInside, enteredCurrent := helpers.EnterObservedConstrainedSpan(lm, generated);
        generated := enteredGenerated;
        insideConstrainedOut := enteredInside;
        currentConstrainedOut := enteredCurrent;
        cueArmed := false;
      }
    }
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  } else {
    var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
    var constrainedPrompt := prompt + stablePrefix;
    var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
    if validCount <= narrowThreshold {
      var next := helpers.AdaptiveConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, narrowThreshold, eosToken
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
    } else {
      var remaining: nat := maxSteps - steps;
      var symbolBudget: nat := if stepTokenBudget == 0 || stepTokenBudget > remaining then remaining else stepTokenBudget;
      var symbolGenerated, symbolCurrent, hitEos, usedSteps := helpers.ConstrainedSymbolInGenerated(
        lm, parser, constrainedPrompt, generated, currentConstrainedOut, symbolBudget, eosToken
      );
      generated := symbolGenerated;
      currentConstrainedOut := symbolCurrent;
      steps := steps + usedSteps;
      if hitEos {
        break;
      }
    }
  }
}

cost := steps;
  }
}
