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
// FIXTURE STRATEGY (not synthesized; a hand-written test fixture for a PURE
// re-eval). Purpose: confirm end-to-end that the grounding-rollback recurrence
// penalty (Change 3) fires on the real 1.5B model.
//
// v2 fix: v1 used the bare `next == "<<"` check to open the span, which NEVER fires
// because the tokenizer splits "<<" into two "<" tokens (the span-entry tokenization
// bug) -- so the span never opened, the grounding phase was never reached, and the
// run was just unconstrained generation (Contains <<>> : no). v2 opens the span
// DETERMINISTICALLY with OpenConstrainedSpan so the grounding phase always runs.
//
// Three phases:
//   1. Open the span deterministically (force "<<") so Phase 2 always executes.
//   2. ONE grounding pass via RegenerateUnitOnGroundingFailure, entered only when
//      the full worst-case budget (maxRetries+1)*maxStepsPerUnit fits in the
//      remaining steps. Its rollback branch calls lm.PenalizeTriedTokenAt, so a
//      grounded=False unit produces a "[recurrence] penalize" log line and the
//      regen diverges. steps is advanced by that whole worst-case bound so the
//      length postcondition |generated| <= |generatedPrefix| + maxSteps is provable.
//   3. Close the span with CloseSpanWithinBudget inside the remaining budget.
// Composes only already-verified library primitives; adds no new decode behavior.
// CSD_RATIONALE_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Generate a single valid SQL query. Output format: SQL: <<your SQL query here>>. Use only tables and columns from the provided schema. Do not include explanation or markdown.";
helpers.AppendTaskGuidance(lm, guidance);

var maxStepsPerUnit: nat := 20;
var maxRetries: nat := 3;
var maxRollbackBudget: nat := 15;
var groundBound: nat := (maxRetries + 1) * maxStepsPerUnit;

var steps: nat := 0;

// Phase 1: open the span deterministically (the bare next=="<<" check never fires
// on this tokenizer). Guarded so steps stays <= maxSteps for arbitrary maxSteps.
if !insideConstrainedOut && steps < maxSteps {
  var gOut, insideO, curO := helpers.OpenConstrainedSpan(lm, generated);
  generated := gOut;
  insideConstrainedOut := insideO;
  currentConstrainedOut := curO;
  steps := steps + 1;
}

// Phase 2: one grounding pass (rollback -> PenalizeTriedTokenAt). Entered only when
// the full worst-case budget fits, so the length bound is provable by construction.
if insideConstrainedOut && steps + groundBound <= maxSteps {
  var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
  var resultConstrained := helpers.RegenerateUnitOnGroundingFailure(
    lm, parser, constrainedPrompt, currentConstrainedOut, eosToken,
    maxStepsPerUnit, maxRetries, maxRollbackBudget
  );
  var stableLen := |generated| - |currentConstrainedOut|;
  var stablePrefix := generated[..stableLen];
  generated := stablePrefix + resultConstrained;
  currentConstrainedOut := resultConstrained;
  steps := steps + groundBound;
}

// Phase 3: close the span within the remaining budget.
if insideConstrainedOut && steps < maxSteps {
  var remain: nat := maxSteps - steps;
  var cg, ci, cc := helpers.CloseSpanWithinBudget(
    lm, parser, prompt, generated, currentConstrainedOut, eosToken, remain
  );
  generated := cg;
  insideConstrainedOut := ci;
  currentConstrainedOut := cc;
  steps := maxSteps;
}

cost := steps;

  }
}
