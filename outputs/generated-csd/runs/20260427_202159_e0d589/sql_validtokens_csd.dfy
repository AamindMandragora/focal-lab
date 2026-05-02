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
    validTokens: seq<Token>,
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
// Spider text-to-SQL CSD with schema-aware semantic steering. The strategy
// tracks whether generation is currently inside a constrained SQL span and, if
// so, the exact constrained prefix currently being validated by the SQL parser.
// It also tracks a lightweight semantic context extracted after the SQL keyword
// "FROM", which approximates the currently mentioned table scope.
//
// Outside constrained spans, generation is free-form so the model can emit the
// delimiter and any surrounding natural text. Inside constrained spans, the
// strategy chooses among three actions based on parser observations: close the
// span immediately when the SQL prefix is complete; use a hard constrained step
// when the grammar is narrow or near a dead end; otherwise perform a masked
// sample after boosting parser-valid candidates that overlap both the caller-
// supplied validTokens and the tracked FROM-context.
//
// The tracked constrained prefix guarantees parser-valid SQL growth, while the
// suffix relation between generated and currentConstrainedOut ensures that the
// parser state always corresponds to the tail of the emitted answer. The extra
// semantic context does not affect soundness, but helps bias choices toward
// schema-relevant continuations for Spider-style SQL generation.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained branch, we only enter constrained mode
//   when next == "<<", and then set currentConstrainedOut := [], which is valid
//   by precondition. In the complete-prefix branch, CloseConstrainedSpan sets
//   insideConstrainedOut to false, so the implication is vacuous. In the hard
//   constrained branch, ConstrainedStep returns a parser-valid next token (or
//   EOS, which breaks), and AppendConstrainedToken preserves validity. In the
//   soft masked branch, MaskValidNextAndEos ensures any non-EOS sampled token
//   is parser-valid, and AppendConstrainedToken preserves validity.
// suffix: In the unconstrained branch, when we open a span we set
//   currentConstrainedOut := [], so the length-0 suffix of generated matches.
//   In the close branch, CloseConstrainedSpan appends the delimiter and resets
//   currentConstrainedOut to [], making the implication vacuous. In both inside-
//   constrained token-appending branches, AppendConstrainedToken appends the
//   same token to generated and currentConstrainedOut, preserving the suffix
//   equality. Semantic-context updates only read state and do not affect it.
// cost accounting: We return cost := steps, so the loop only needs to maintain
//   0 <= steps <= maxSteps. UnconstrainedStep, CloseConstrainedSpan, the hard
//   constrained branch, and the soft masked branch each consume exactly one
//   decoding step and increment steps by 1; pure helper queries and boosts do
//   not affect steps. Therefore cost == steps at return and cost <= maxSteps.
// progress: UnconstrainedStep appends exactly one token and we do steps :=
//   steps + 1. CloseConstrainedSpan appends one delimiter token and also
//   increments steps by 1. The hard constrained branch appends one token via
//   AppendConstrainedToken after one ConstrainedStep and increments steps by 1.
//   The soft masked branch samples one token after masking, appends at most one
//   token, and increments steps by 1. Break branches stop immediately, so
//   |generated| <= |generatedPrefix| + steps is preserved throughout.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var fromKeyword: Token := "FROM";
var narrowThreshold: nat := 12;
var semanticContext: seq<Token> := [];
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
    semanticContext := helpers.ExtractAfterKeyword(currentConstrainedOut, fromKeyword);
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
      var narrow := helpers.DeadEndDetection(parser, currentConstrainedOut, narrowThreshold);
      if narrow {
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

        var candidates := helpers.TopValidCandidates(
          lm, parser, constrainedPrompt, currentConstrainedOut, 24, eosToken
        );

        if |validTokens| > 0 {
          var preferred := helpers.IntersectTokenSets(candidates, validTokens);
          if |preferred| > 0 {
            helpers.BoostTokenLogits(lm, preferred, 5.0);
          }
        }

        if |semanticContext| > 0 {
          var scoped := helpers.IntersectTokenSets(candidates, semanticContext);
          if |scoped| > 0 {
            helpers.BoostTokenLogits(lm, scoped, 6.0);
          }
        }

        lm.MaskValidNextAndEos(parser, currentConstrainedOut, eosToken);
        var next := lm.ChooseNextToken();
        helpers.cost := helpers.cost + 1;
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
    if maxSteps > 0 && cost == 0 { cost := 1; }  // guarantee progress postcondition
  }
}
