// CSD_RATIONALE_BEGIN
// Immediate-visible-span GSM-symbolic CSD. The strategy appends task guidance
// asking the LM to solve step by step and keep every symbolic expression inside
// visible << >> delimiters, then forces entry into a constrained span whenever
// generation is outside one. Inside a span, it uses parser-constrained symbol
// generation, so arithmetic-expression content remains parser-valid. As soon as
// the parser reports a complete constrained expression, the strategy closes the
// visible span with >> and returns.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: AppendTaskGuidance is prompt-only and does not change the
//   constrained state. In the outside branch, OpenConstrainedSpan sets
//   insideConstrainedOut to true and currentConstrainedOut to [], which is valid
//   by parser.IsValidPrefix([]). In the complete-prefix branch,
//   CloseConstrainedSpan exits constrained mode and clears the current
//   constrained prefix, so the implication is vacuous. In the symbol branch,
//   ConstrainedSymbolInGenerated returns a parser-valid constrained prefix;
//   since insideConstrainedOut remains true, the invariant is preserved.
// progress: AppendTaskGuidance consumes no token budget and appends no visible
//   output. OpenConstrainedSpan and CloseConstrainedSpan each consume one
//   token-step and append at most one visible delimiter, so incrementing steps
//   by 1 preserves |generated| <= |generatedPrefix| + steps. The symbol branch
//   increments steps by stepsUsed, the token budget consumed by
//   ConstrainedSymbolInGenerated; visible growth is at most that consumed
//   budget, even if EOS or rejected suffix tokens are consumed without being
//   appended, so the output-length bound is preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm,
  "Solve the math word problem step by step. Wrap every intermediate symbolic arithmetic expression and the final answer inside visible << >> delimiters. Inside each span write only a valid arithmetic expression: no prose, no units, no Markdown, no LaTeX, and no placeholder text. Preserve variable names exactly as given, including underscores. Use all relevant quantities from the problem.");

var steps: nat := 0;

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
    var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);
    generated := openedGenerated;
    insideConstrainedOut := openedInside;
    currentConstrainedOut := openedCurrent;
    steps := steps + 1;
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
    break;
  } else {
    var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
    var constrainedPrompt := prompt + stablePrefix;
    var symbolBudget: nat := maxSteps - steps;
    var symbolGenerated, symbolOut, hitEos, stepsUsed := helpers.ConstrainedSymbolInGenerated(
      lm, parser, constrainedPrompt, generated, currentConstrainedOut, symbolBudget, eosToken
    );
    generated := symbolGenerated;
    currentConstrainedOut := symbolOut;
    steps := steps + stepsUsed;
    if hitEos {
      break;
    }
  }
}

cost := steps;
