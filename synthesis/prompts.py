"""
Prompt templates for Qwen-based CSD strategy generation.

This project synthesizes *constrained decoding strategies* (CSDs), not the final
task output itself. The LLM should choose an appropriate verified strategy
primitive and parameters based on the *use-case* described by the task.

The generator expects these entrypoints:
- build_initial_prompt(task_description)
- build_verification_error_prompt(task_description, previous_strategy, error_message)
- build_runtime_error_prompt(previous_strategy, error_traceback)
- build_compilation_error_prompt(previous_strategy, error_message)
- build_format_repair_prompt(previous_strategy)
"""

# NOTE:
# The synthesized output is injected into `dafny/GeneratedCSD.dfy` as the BODY
# of method `MyCSDStrategy(...)`.
#
# The output is a multi-line Dafny method body.
# It receives the full generated prefix so far plus explicit state for the
# currently active constrained segment, if any.

SYSTEM_PROMPT = """\
You are generating a *constrained decoding strategy* (CSD) that composes verified tools to produce valid output.

You must output ONLY the Dafny method body for:

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
    requires eosToken in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    ensures cost <= maxSteps
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained

## Output rules
- Start with: `// CSD_RATIONALE_BEGIN\n// ...\n// CSD_RATIONALE_END`
- Immediately after the rationale, emit a proof sketch block:
  `// CSD_PROOF_SKETCH_BEGIN\n// ...\n// CSD_PROOF_SKETCH_END`.
  For each of the four non-trivial loop invariants (parser validity, suffix
  relationship, cost accounting, progress bound), briefly explain in the
  sketch why every branch of your loop body preserves the invariant. See the
  "## Proof sketch discipline" section below for details.
- Initialize all out-parameters before any loop/recursion.
- Assign `cost` before returning.
- Do NOT redeclare out-parameters as locals.
- Use the provided `helpers` instance (type `CSDHelpers`). Do NOT write `var helpers := new CSDHelpers();`.
- Do NOT use `CSDHelpers.<Method>` for instance methods.
- `Token` is type `string`. EOS check: `if next == eosToken`. No string conversion needed.
- EOS is terminal. If `next == eosToken`, stop generation immediately with `break` or by making the loop guard false; never ignore `eosToken` and continue generating.
- Use a local `steps` counter for loop progress, decreases, and returned cost accounting. Prefer `cost := steps` before returning; do NOT rely on `helpers.cost` for the returned `cost`.
- Logit-adjustment amounts must be `real` values like `3.0`, `8.0`, or `100.0`, not integer literals like `3` or `100`.
- Never call `helpers.CloseConstrainedSpan(...)` only because a sampled token looks like `">>"`; call it only from a branch where `parser.IsCompletePrefix(currentConstrainedOut)` is already known to hold.
- `prompt` is type `Prefix` (= `seq<Token>`). Sequence concat uses `+`, NOT `++`.
- `Contains(token, "<<")` is a module-level function — NOT `helpers.Contains`.
- `generated` / `generatedPrefix` contain the full answer text, including any `<<` and `>>` delimiter tokens.
- `currentConstrained` / `currentConstrainedOut` track only the active constrained segment contents between delimiters, never the delimiter tokens themselves.

## Dafny syntax rules
- Multi-return: `var a, b := helper(...)` — NOT `var (a, b) :=`
- `_` is NOT a discard — use a named variable
- Method calls CANNOT appear in `if`/`while`/`decreases`. Assign to variable first.
- `decreases` only uses variables/arithmetic — no method calls.
- `continue` does NOT exist. Use nested `if/else`.
- Variable init uses `:=` NEVER `=`.
- Invariants/decreases go BETWEEN condition and `{{`.
- NEVER call `lm.GenerateLogits` before a helper — helpers call it internally.
- NEVER manually increment `helpers.cost` after calling a helper.

## Available Tools

### Runtime inputs
The method receives validTokenGroups (type seq<seq<Token>>) as a parameter
alongside lm, parser, etc. It is a sequence of token groups supplied by the
caller; its contents are determined externally to the strategy. The strategy
may read or ignore it. Type only: seq<seq<Token>> (where Token = seq<CodePoint>).
No precondition or postcondition is attached to its contents — it can be empty,
inner groups can be empty, groups may overlap, and groups may contain tokens
not in lm.Tokens.

### Raw logit manipulation
```
lm.GenerateLogits(prompt + generated);
helpers.BoostTokenLogits(lm, tokens, amount);
helpers.PenalizeTokenLogits(lm, tokens, amount);
var topToken := helpers.GetHighestLogitToken(lm);
var logit := helpers.GetTokenLogit(lm, token);
helpers.ScaleAllLogits(lm, scalar);
lm.MaskValidNextAndEos(parser, prefix, eosToken);
var next := lm.ChooseNextToken();
var next := lm.ChooseNextTokenUnconstrained();
helpers.cost := helpers.cost + 1;
```

### Helper methods
```
var next := helpers.UnconstrainedStep(lm, prompt, generated);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.OpenConstrainedSpan(lm, generated);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrained, next);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrained);
var generatedOut, insideOut, currentOut, hitEos := helpers.ConstrainedStep(lm, parser, constrainedPrompt, generated, currentConstrained, eosToken);
var next := helpers.ConstrainedSample(lm, parser, currentConstrained, eosToken);
var next := helpers.GroupBoostedConstrainedStep(lm, parser, prompt + stablePrefix, currentConstrained, validTokenGroups, 4.0, eosToken);
var next := helpers.AdaptiveConstrainedStep(lm, parser, prompt + stablePrefix, currentConstrained, validTokenGroups, 4.0, 12, eosToken);
var next := helpers.PenalizedConstrainedStep(lm, parser, prompt + stablePrefix, currentConstrained, tokensToPenalize, 5.0, eosToken);
var generatedOut, stoppedOnOpenSpan, stoppedOnEos, stepsUsed := helpers.UnconstrainedChunk(lm, prompt, generated, maxChunkTokens, openSpanToken, eosToken);
var currentOut, hitEos, stepsUsed := helpers.ConstrainedSymbol(lm, parser, constrainedPrompt, currentConstrained, stepTokenBudget, eosToken);
```

`OpenConstrainedSpan(lm: LM, generated: Prefix)
  returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)`
  Requires: `"<<" in lm.Tokens`.
  Ensures: `generatedOut == generated + ["<<"]`, `|generatedOut| == |generated| + 1`,
  `insideOut == true`, `currentOut == []`. Cost: 1.

`AppendConstrainedToken(lm: LM, parser: Parser, generated: Prefix,
  currentConstrained: Prefix, next: Token)
  returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)`
  Requires: `parser.IsValidPrefix(currentConstrained)`,
  `parser.IsValidPrefix(currentConstrained + [next])`.
  Ensures: `generatedOut == generated + [next]`, `|generatedOut| == |generated| + 1`,
  `currentOut == currentConstrained + [next]`, `insideOut == true`. Cost: 0.

`ConstrainedStep(lm: LM, parser: Parser, prompt: Prefix,
  generated: Prefix, currentConstrained: Prefix, eosToken: Token)
  returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix, hitEos: bool)`
  Requires: `lm.ValidTokensIdsLogits()`, `parser.IsValidPrefix(currentConstrained)`,
  suffix invariant `generated[|generated| - |currentConstrained|..] == currentConstrained`.
  Ensures on `hitEos`: state unchanged (`generatedOut == generated`, etc.).
  Ensures on `!hitEos`: `|generatedOut| == |generated| + 1`,
  `parser.IsValidPrefix(currentOut)`, suffix invariant preserved. `insideOut == true`.
  Generates logits, masks to parser-valid tokens + EOS, samples. On non-EOS,
  appends the token to both `generated` and `currentConstrained` and returns
  updated state. On EOS, returns state unchanged with `hitEos == true`. Cost: 1.

`ConstrainedSample(lm: LM, parser: Parser, prefix: Prefix, eosToken: Token)
  returns (next: Token)`
  Requires: `lm.ValidTokensIdsLogits()`, `parser.IsValidPrefix(prefix)`.
  Ensures: `next == eosToken || parser.IsValidPrefix(prefix + [next])`.
  Masks logits to parser-valid tokens + EOS, then samples. Does NOT call
  `GenerateLogits` — caller must populate logits first. Use when you need
  custom logit manipulation (boost/penalize) before constrained sampling.
  Cost: 1.

`GroupBoostedConstrainedStep(lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
  groups: seq<seq<Token>>, boostAmount: real, eosToken: Token)
  returns (next: Token)`
  Requires: `lm.ValidTokensIdsLogits()`, `parser.IsValidPrefix(constrainedPrefix)`,
  `boostAmount >= 0.0 && boostAmount <= 1e8`.
  Ensures: `next == eosToken || parser.IsValidPrefix(constrainedPrefix + [next])`.
  All-in-one constrained step: calls `GenerateLogits(prompt + constrainedPrefix)`,
  then `BoostValidGroups` on the given groups (if non-empty), then masks to
  parser-valid tokens + EOS, then samples. Equivalent to calling GenerateLogits +
  BoostValidGroups + MaskValidNextAndEos + ChooseNextToken manually, but in a
  single verified helper. Cost: 1.

`AdaptiveConstrainedStep(lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
  groups: seq<seq<Token>>, boostAmount: real, narrowThreshold: nat, eosToken: Token)
  returns (next: Token)`
  Requires: `lm.ValidTokensIdsLogits()`, `parser.IsValidPrefix(constrainedPrefix)`,
  `boostAmount >= 0.0 && boostAmount <= 1e8`.
  Ensures: `next == eosToken || parser.IsValidPrefix(constrainedPrefix + [next])`.
  Generates logits, then checks how many parser-valid next tokens exist. If the
  count is at or below `narrowThreshold`, boosts the given token groups (via
  `BoostValidGroups`); otherwise skips boosting. Then masks to parser-valid
  tokens + EOS and samples. Boosting is skipped when the number of
  parser-valid next tokens exceeds `narrowThreshold`. Cost: 1.

`PenalizedConstrainedStep(lm: LM, parser: Parser, prompt: Prefix, constrainedPrefix: Prefix,
  tokensToPenalize: seq<Token>, penaltyAmount: real, eosToken: Token)
  returns (next: Token)`
  Requires: `lm.ValidTokensIdsLogits()`, `parser.IsValidPrefix(constrainedPrefix)`,
  `penaltyAmount >= 0.0 && penaltyAmount <= 1e8`,
  all tokens in `tokensToPenalize` must be in `lm.Tokens`.
  Ensures: `next == eosToken || parser.IsValidPrefix(constrainedPrefix + [next])`.
  All-in-one constrained step with logit penalties: calls
  `GenerateLogits(prompt + constrainedPrefix)`, then `PenalizeTokenLogits`
  on the specified tokens (subtracts `penaltyAmount` from their logits),
  then masks to parser-valid tokens + EOS and samples. Useful for
  discouraging specific tokens at particular grammar
  positions. Cost: 1.

`CloseConstrainedSpan(lm: LM, parser: Parser, generated: Prefix,
  currentConstrained: Prefix)
  returns (generatedOut: Prefix, insideOut: bool, currentOut: Prefix)`
  Requires: `parser.IsCompletePrefix(currentConstrained)`, `">>" in lm.Tokens`.
  Ensures: `|generatedOut| <= |generated| + 1`, `insideOut == false`,
  `currentOut == []`. Cost: 1.

### Parser queries
```
var narrow := helpers.DeadEndDetection(parser, currentConstrained, minValidCount);
var count := helpers.ValidTokenCount(parser, currentConstrained);
var valid := helpers.IsTokenValidNext(parser, currentConstrained, token);
var candidates := helpers.TopValidCandidates(lm, parser, prompt, currentConstrained, maxCandidates, eosToken);
var rolled := helpers.RollbackToValidPrefix(parser, constrainedPrefix);
var generatedOut, currentOut := helpers.RollbackConstrainedSpan(parser, stablePrefix, generated, currentConstrained);
var following := helpers.ExtractAfterKeyword(prefix, keyword);
var intersection := helpers.IntersectTokenSets(a, b);
var difference := helpers.SubtractTokenSets(a, b);
var flat := helpers.FlattenTokenGroups(groups);
var idx := helpers.GroupContaining(groups, token);
var rolled := helpers.RollbackToBoundary(parser, currentConstrained, boundaryToken);
var rolled, excludedTok, hasExcl := helpers.RollbackAndExclude(parser, currentConstrained, boundaryToken);
var tok, found := helpers.LastTokenBefore(s, sep);
var anyValid := helpers.GroupHasValidMember(parser, prefix, group);
helpers.BoostValidGroups(lm, parser, prefix, groups, amount);
```

`FlattenTokenGroups(groups: seq<seq<Token>>) returns (flat: seq<Token>)`
  Ensures: every token in `flat` came from at least one inner group.

`GroupContaining(groups: seq<seq<Token>>, tok: Token) returns (idx: int)`
  Ensures: `-1 <= idx < |groups|`, and if `idx >= 0` then `tok in groups[idx]`.

`RollbackToBoundary(parser: Parser, currentConstrained: Prefix, boundaryToken: Token)
  returns (repaired: Prefix)`
  Requires: `parser.IsValidPrefix([])`.
  Ensures: `parser.IsValidPrefix(repaired) && |repaired| <= |currentConstrained|`.
  Ensures: `repaired == currentConstrained[..|repaired|]` (i.e. `repaired` is a
  prefix of `currentConstrained`, so callers can re-sync `generated` by
  trimming `|currentConstrained| - |repaired|` tokens off its end).
  Walks back through `currentConstrained` until just past the last occurrence
  of `boundaryToken`, or returns `[]` if absent.

`RollbackAndExclude(parser: Parser, currentConstrained: Prefix, boundaryToken: Token)
  returns (repaired: Prefix, excludedToken: Token, hasExcluded: bool)`
  Requires: `parser.IsValidPrefix([])`.
  Ensures: `parser.IsValidPrefix(repaired) && |repaired| <= |currentConstrained|`.
  Ensures: `repaired == currentConstrained[..|repaired|]`.
  Ensures: if `hasExcluded` then `|repaired| < |currentConstrained|` and
  `excludedToken == currentConstrained[|repaired|]` — the first token after
  the rollback point in the original prefix (the divergence token).
  Like `RollbackToBoundary` but also identifies the token that started the
  path leading to the dead end. Callers can penalize or mask `excludedToken`
  before re-generating to avoid repeating the same path.

`LastTokenBefore(s: Prefix, sep: Token) returns (tok: Token, found: bool)`
  Ensures: if `found` then `tok` is the token immediately preceding the LAST
  occurrence of `sep` in `s` (at some index `i >= 1` where `s[i] == sep`,
  `tok == s[i-1]`). If `sep` is absent or only appears at index 0, `found` is
  false. Pure — no parser interaction, no LM mutation.

`GroupHasValidMember(parser: Parser, prefix: Prefix, group: seq<Token>)
  returns (anyValid: bool)`
  Requires: `parser.IsValidPrefix(prefix)`.
  Pure query — does NOT modify `lm.Logits`.
  Ensures: `anyValid <==> (exists t :: t in group && parser.ValidNextToken(prefix, t))`.
  Cost: O(|group|) per-token DFA queries.

`BoostValidGroups(lm: LM, parser: Parser, prefix: Prefix, groups: seq<seq<Token>>,
  amount: real)`
  Requires: `lm.ValidTokensIdsLogits()`, `parser.IsValidPrefix(prefix)`,
  `amount >= 0.0 && amount <= 1e8`.
  Ensures: `lm.ValidTokensIdsLogits()`.
  For each group in `groups`, checks if any member is parser-valid at `prefix`;
  if so, boosts all tokens in that group by `amount`. Replaces a manual
  `while i < |groups| { GroupHasValidMember + BoostTokenLogits }` loop.

## Proof sketch discipline

Your output must include a `// CSD_PROOF_SKETCH_BEGIN ... // CSD_PROOF_SKETCH_END`
block between the rationale and the method body.

For each of the following three non-trivial loop invariants, explain in one or
two sentences per branch why that branch preserves the invariant:

1. `parser_validity`:
   `insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)`
2. `suffix`:
   `insideConstrainedOut ==> generated[|generated| - |currentConstrainedOut|..] == currentConstrainedOut`
3. `progress`:
   `|generated| <= |generatedPrefix| + steps`
   The `decreases maxSteps - steps` annotation requires every non-breaking
   branch of the loop body to strictly increase `steps`. Branches fall into
   one of these patterns:
   - appends exactly 1 token and increments `steps` by 1 (UnconstrainedStep,
     CloseConstrainedSpan, ConstrainedStep, manual
     mask+sample+append),
   - appends `stepsUsed` tokens and increments `steps` by `stepsUsed`
     (UnconstrainedChunk, ConstrainedSymbol),
   - appends no tokens but still increments `steps` by at least 1 (e.g. a
     pure recovery / query-only branch that calls `RollbackToBoundary`,
     `RollbackToValidPrefix`, `RollbackConstrainedSpan`, or any sequence of
     non-bumping helpers — these helpers do not advance `steps` on their own,
     so the branch must do `steps := steps + 1` itself),
   - breaks immediately.
   All arithmetic is linear; `|generated|` only ever stays the same or shrinks
   in the no-append patterns, so the bound is preserved trivially there.

This is a PROVABILITY discipline, not a SIMPLICITY discipline. Novel, creative
strategies are encouraged — as long as you can articulate a coherent
preservation argument for each invariant in each branch you introduce. If you
CANNOT write that argument for some branch, that branch is broken; fix the
design before emitting Dafny. Do NOT replace a creative strategy with a
CRANE-shaped one just to make the sketch shorter.

Helper bookkeeping you can rely on in the sketch:
- Cost-bumping helpers (bump `helpers.cost` by 1 internally):
  `UnconstrainedStep`, `ConstrainedStep`, `ConstrainedSample`, `BoostedConstrainedStep`, `GroupBoostedConstrainedStep`, `AdaptiveConstrainedStep`, `PenalizedConstrainedStep`, `CloseConstrainedSpan`, `OpenConstrainedSpan`.
- Cost-bumping helpers (bump `helpers.cost` by `stepsUsed` internally):
  `UnconstrainedChunk` (bumps by the returned `stepsUsed`).
- Non-bumping helpers (pure state shuffling or query — do NOT advance
  `steps` on their own; if used in a branch that does not call any
  cost-bumping helper, the branch must `steps := steps + 1` itself or break):
  `AppendConstrainedToken`, `GetHighestLogitToken`, `GetTokenLogit`,
  `TopValidCandidates`, `IsTokenValidNext`, `DeadEndDetection`, `ValidTokenCount`,
  `RollbackToValidPrefix`, `RollbackConstrainedSpan`, `RollbackToBoundary`,
  `LastTokenBefore`, `GroupHasValidMember`, `BoostValidGroups`, `FlattenTokenGroups`,
  `GroupContaining`, `IntersectTokenSets`, `SubtractTokenSets`,
  `ExtractAfterKeyword`, `BoostTokenLogits`, `PenalizeTokenLogits`,
  `ScaleAllLogits`.
- Primitive `lm.ChooseNextToken()` / `lm.ChooseNextTokenUnconstrained()`: you
  must manually do `helpers.cost := helpers.cost + 1` after these.
- `lm.GenerateLogits(...)`: does not bump cost (it just populates logits).

"""

INITIAL_GENERATION_PROMPT = """\
Generate a complete Dafny method body for this use-case.

Task:
{task_description}

Output ONLY the Dafny method body. Do NOT output a method signature, outer wrapper text, or markdown code fences.

## Verified Examples

```dafny
// CSD_RATIONALE_BEGIN
// Simple delimiter-triggered CSD. Generate freely until "<<" appears, then
// constrain until the parser says the span can close.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained branch we only flip insideConstrainedOut
//   to true when next == "<<", and we set currentConstrainedOut := [] which is
//   a valid prefix. In the complete-prefix branch, CloseConstrainedSpan flips
//   insideConstrainedOut to false, making the implication vacuous. In the
//   ConstrainedStep branch, the helper returns parser-valid updated state
//   directly (or EOS, which breaks).
// suffix: In the unconstrained branch, currentConstrainedOut is [] whenever
//   insideConstrainedOut is true, and the length-0 suffix of generated matches
//   []. In the complete-prefix branch, CloseConstrainedSpan atomically appends
//   ">>" to generated and resets currentConstrainedOut to []. In the constrained
//   branch, ConstrainedStep maintains the suffix invariant internally.
// progress: Every branch appends at most one token to generated and steps grows
//   by 1, so |generated| - |generatedPrefix| <= steps <= steps * stepTokenBudget.
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
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var g, i, c, hitEos := helpers.ConstrainedStep(lm, parser, constrainedPrompt, generated, currentConstrainedOut, eosToken);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var valid := helpers.IsTokenValidNext(parser, currentConstrainedOut, next);
      if valid {{
        var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := appendedGenerated;
        insideConstrainedOut := appendedInside;
        currentConstrainedOut := appendedCurrent;
      }}
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Boost-and-sample soft-constrained CSD. Inside a span, rather than hard-
// constraining to parser-valid tokens, we boost the logits of the top-k valid
// candidates and then sample unconstrained. If the sampled token is valid we
// append it; if not, we skip and re-draw next iteration. This keeps the
// distribution closer to the model's unconstrained preferences while still
// biasing toward valid continuations.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Enter only via the unconstrained branch when next == "<<",
//   at which point currentConstrainedOut := [] (valid). Complete-prefix branch
//   flips insideConstrainedOut to false via CloseConstrainedSpan. Inside the
//   span, AppendConstrainedToken is only called when IsTokenValidNext(next)
//   holds; the !valid branch is a skip that does not modify state.
// suffix: Unchanged in skip; in append, same token added to both generated and
//   currentConstrainedOut; CloseConstrainedSpan atomically appends ">>" and
//   resets currentConstrainedOut to [].
// progress: Every branch appends at most one token; attempts grows by 1, so
//   |generated| <= |generatedPrefix| + attempts * stepTokenBudget holds.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var attempts: nat := 0;

while attempts < maxSteps
  invariant 0 <= attempts <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant insideConstrainedOut ==> generated[|generated| - |currentConstrainedOut|..] == currentConstrainedOut
  invariant |generated| <= |generatedPrefix| + attempts
  invariant cost == 0
  decreases maxSteps - attempts
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    attempts := attempts + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    attempts := attempts + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var candidates := helpers.TopValidCandidates(
      lm, parser, constrainedPrompt, currentConstrainedOut, 4, eosToken
    );
    lm.GenerateLogits(constrainedPrompt + currentConstrainedOut);
    helpers.BoostTokenLogits(lm, candidates, 4.0);
    var next := lm.ChooseNextTokenUnconstrained();
    helpers.cost := helpers.cost + 1;
    attempts := attempts + 1;
    if next == eosToken {{
      break;
    }} else {{
      var valid := helpers.IsTokenValidNext(parser, currentConstrainedOut, next);
      if valid {{
        var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := appendedGenerated;
        insideConstrainedOut := appendedInside;
        currentConstrainedOut := appendedCurrent;
      }}
      // If !valid, skip this step — equivalent to rejection sampling; the next
      // iteration will re-draw with a fresh logit distribution.
    }}
  }}
}}

cost := attempts;
```

```dafny
// CSD_RATIONALE_BEGIN
// Recursive CSD. One step of work per invocation, then tail-recurse with a
// smaller maxSteps. Inside a span we use TopValidCandidates to pick a
// constrained token directly (no sampling noise).
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// This strategy is recursive rather than loop-based, so the sketch covers the
// four non-trivial postconditions at each return path.
// parser_validity (ensures insideConstrainedOut ==> IsValidPrefix(currentConstrainedOut)):
//   maxSteps==0 base case returns the inputs unchanged, and the precondition
//   already guarantees the invariant. In the non-constrained recursive branch
//   we call the recursion with currentConstrained = [] (valid) when entering a
//   span, or with the existing valid prefix otherwise. In the complete-prefix
//   branch, CloseConstrainedSpan returns closedInside = false (implication
//   vacuous) before we recurse. In the TopValidCandidates branch, chosen is
//   valid-next by construction of TopValidCandidates, so AppendConstrainedToken
//   yields a valid prefix; recursion preserves it.
// suffix (ensures the suffix equality): base case unchanged. Every helper we
//   call (UnconstrainedStep + manual append, CloseConstrainedSpan, Append)
//   respects the suffix relation; recursion inherits it.
// cost (ensures cost <= maxSteps): each recursive call consumes exactly one
//   cost-bumping helper (UnconstrainedStep, CloseConstrainedSpan, or
//   AppendConstrainedToken combined with the implicit TopValidCandidates
//   accounting — here we rely on helpers.cost being tracked by the helpers
//   themselves) and recurses with maxSteps - 1. We return
//   cost := helpers.cost + subCost; induction on maxSteps gives cost <= maxSteps.
// progress (ensures |generated| <= |generatedPrefix| + maxSteps): each call
//   appends at most one token before recursing with maxSteps - 1.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

if maxSteps == 0 {{
}} else if !insideConstrained {{
  var next := helpers.UnconstrainedStep(lm, prompt, generatedPrefix);
  if next == eosToken {{
    cost := helpers.cost;
  }} else {{
    var nextGenerated := generatedPrefix + [next];
    if next == "<<" {{
      var subGenerated, subInside, subCurrent, subCost := MyCSDStrategy(
        lm, parser, prompt, nextGenerated, true, [], maxSteps - 1, eosToken
      );
      generated := subGenerated;
      insideConstrainedOut := subInside;
      currentConstrainedOut := subCurrent;
      cost := helpers.cost + subCost;
    }} else {{
      var subGenerated, subInside, subCurrent, subCost := MyCSDStrategy(
        lm, parser, prompt, nextGenerated, false, [], maxSteps - 1, eosToken
      );
      generated := subGenerated;
      insideConstrainedOut := subInside;
      currentConstrainedOut := subCurrent;
      cost := helpers.cost + subCost;
    }}
  }}
}} else if parser.IsCompletePrefix(currentConstrained) {{
  var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
    lm, parser, generatedPrefix, currentConstrained
  );
  var subGenerated, subInside, subCurrent, subCost := MyCSDStrategy(
    lm, parser, prompt, closedGenerated, closedInside, closedCurrent, maxSteps - 1, eosToken
  );
  generated := subGenerated;
  insideConstrainedOut := subInside;
  currentConstrainedOut := subCurrent;
  cost := helpers.cost + subCost;
}} else {{
  var constrainedPrompt := prompt + generatedPrefix[..|generatedPrefix| - |currentConstrained|];
  var candidates := helpers.TopValidCandidates(
    lm, parser, constrainedPrompt, currentConstrained, 2, eosToken
  );
  var chosen := candidates[0];
  if chosen == eosToken && |candidates| > 1 {{
    chosen := candidates[1];
  }}

  if chosen == eosToken {{
    cost := helpers.cost;
  }} else {{
    var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
      lm, parser, generatedPrefix, currentConstrained, chosen
    );
    var subGenerated, subInside, subCurrent, subCost := MyCSDStrategy(
      lm, parser, prompt, appendedGenerated, appendedInside, appendedCurrent, maxSteps - 1, eosToken
    );
    generated := subGenerated;
    insideConstrainedOut := subInside;
    currentConstrainedOut := subCurrent;
    cost := helpers.cost + subCost;
  }}
}}
```

```dafny
// CSD_RATIONALE_BEGIN
// Chunked-outside CSD. Outside a constrained span we generate unconstrained
// tokens in a single multi-token call (`UnconstrainedChunk`) that breaks early
// on EOS or on the open-span delimiter `"<<"`. Inside a span we decode token
// by token using the parser the same way the simple delimiter-triggered
// strategy does. Multi-token chunking amortizes per-token dispatch overhead
// across the unconstrained region.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Outside the span, insideConstrainedOut stays false unless
//   UnconstrainedChunk reports stoppedOnOpenSpan, in which case we set
//   currentConstrainedOut := [], a valid prefix by the method precondition
//   parser.IsValidPrefix([]). In the complete-prefix branch, CloseConstrainedSpan
//   flips insideConstrainedOut to false, making the implication vacuous. In the
//   constrained-step branch, AppendConstrainedToken is only invoked after
//   ConstrainedStep returned a parser-valid next token (or EOS, which breaks).
// suffix: Outside the span, |currentConstrainedOut| = 0, so the suffix
//   implication is vacuous when insideConstrainedOut is false. When
//   stoppedOnOpenSpan is true, the chunk's postcondition gives that the last
//   token of generated is "<<"; we set currentConstrainedOut := [], and the
//   length-0 suffix of generated equals []. CloseConstrainedSpan and
//   AppendConstrainedToken preserve the suffix relation as in the simple
//   example.
// progress: Chunk branch: |generatedOut| <= |generated| + stepsUsed and
//   steps := steps + stepsUsed, so |new_generated| <= |generatedPrefix| +
//   steps + stepsUsed = |generatedPrefix| + new_steps. Other branches append
//   ≤1 token and steps += 1. Linear arithmetic throughout. ✓
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
{{
  if !insideConstrainedOut {{
    var chunkBudget: nat := maxSteps - steps;
    var chunkedG, stoppedOpen, stoppedEos, stepsUsed := helpers.UnconstrainedChunk(
      lm, prompt, generated, chunkBudget, "<<", eosToken
    );
    generated := chunkedG;
    steps := steps + stepsUsed;
    if stoppedEos {{
      break;
    }} else if stoppedOpen {{
      insideConstrainedOut := true;
      currentConstrainedOut := [];
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var g, i, c, hitEos := helpers.ConstrainedStep(lm, parser, constrainedPrompt, generated, currentConstrainedOut, eosToken);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Adaptive-narrowness CSD. Inside a constrained span, the strategy uses the
// parser's own narrowness signal to choose between two decoding modes:
// tight (ConstrainedStep) when the grammar allows few continuations and the
// model has little meaningful choice anyway, or loose (ConstrainedSymbol with
// a budget) when the grammar is wide and the model should generate naturally.
//
// This avoids imposing constraint where it hurts: at wide positions the model's
// training distribution is a better guide than token-by-token forcing. At
// narrow positions the grammar already restricts options so tight constraint
// costs almost nothing in terms of distribution shift.
//
// The `narrowThreshold` local constant controls when tight vs loose is used.
// A value around 10-20 works for most grammars: positions with few valid
// continuations are typically semantically constrained; positions with many
// valid continuations are typically structural with many valid options.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: CloseConstrainedSpan makes implication vacuous. In the
//   tight branch, ConstrainedStep returns a parser-valid next token and
//   AppendConstrainedToken preserves validity. In the loose branch,
//   ConstrainedSymbol postcondition guarantees parser.IsValidPrefix(symbolOut).
// suffix: After CloseConstrainedSpan, implication vacuous. In the tight branch
//   AppendConstrainedToken preserves the suffix relation. In the loose branch
//   we set generated := stablePrefix + symbolOut and currentConstrainedOut
//   := symbolOut, so generated[|generated| - |symbolOut|..] == symbolOut. ✓
// progress: Tight branch: steps += 1, |generated| grows by 1 ≤ steps + 1. ✓
//   Loose branch: steps += stepsUsed (≥ 1 by ConstrainedSymbol postcondition),
//   |generated| grows by |symbolOut|-|currentConstrainedOut| ≤ stepsUsed. ✓
//   Both maintain |generated| <= |generatedPrefix| + steps.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var narrowThreshold: nat := 20;
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
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
    if validCount <= narrowThreshold {{
      var g, i, c, hitEos := helpers.ConstrainedStep(lm, parser, constrainedPrompt, generated, currentConstrainedOut, eosToken);
      steps := steps + 1;
      if next == eosToken {{
        break;
      }} else {{
        var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := appendedGenerated;
        insideConstrainedOut := appendedInside;
        currentConstrainedOut := appendedCurrent;
      }}
    }} else {{
      var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
      var symbolOut, hitEos, stepsUsed := helpers.ConstrainedSymbol(
        lm, parser, constrainedPrompt, currentConstrainedOut, stepTokenBudget, eosToken
      );
      generated := stablePrefix + symbolOut;
      currentConstrainedOut := symbolOut;
      steps := steps + stepsUsed;
      if hitEos {{
        break;
      }}
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Context-tracking CSD. Maintains strategy-local semantic context as a
// seq<Token> that accumulates across loop iterations. After each token is
// appended to the constrained span, the strategy queries that span for tokens
// following a task-specific keyword (via ExtractAfterKeyword) to build a
// running set of "semantically in-scope" tokens. At candidate-selection
// positions, it intersects the parser-valid candidate set with this context
// set and boosts the intersection, gently steering the model toward contextually
// relevant choices without hard-forcing any single token.
//
// This is useful whenever a structured output has a "scope-defining" keyword
// that establishes which tokens are semantically valid later (e.g. FROM for
// tables in SQL, LET/:= for variable names in arithmetic, IMPORT for module
// names in code). The strategy discovers and tracks this scope dynamically.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: After MaskValidNextAndEos, any non-EOS token returned by
//   ChooseNextToken is parser-valid (by the mask postcondition); the EOS branch
//   breaks. AppendConstrainedToken preserves validity. CloseConstrainedSpan
//   makes the implication vacuous. The context variable is never passed to the
//   parser, so it cannot affect parser_validity.
// suffix: AppendConstrainedToken and CloseConstrainedSpan maintain the suffix
//   invariant by their postconditions. The context update reads from
//   currentConstrainedOut but does not modify generated or currentConstrainedOut.
// progress: Every branch increments steps by 1 and appends at most one token,
//   so |generated| <= |generatedPrefix| + steps is preserved throughout.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var semanticContext: seq<Token> := [];
var scopeKeyword: Token := "FROM";
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
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    // Update semantic context from accumulated span content
    semanticContext := helpers.ExtractAfterKeyword(currentConstrainedOut, scopeKeyword);

    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    lm.GenerateLogits(constrainedPrompt + currentConstrainedOut);

    // Steer toward contextually relevant tokens when context is non-empty
    if |semanticContext| > 0 {{
      var candidates := helpers.TopValidCandidates(
        lm, parser, constrainedPrompt, currentConstrainedOut, 20, eosToken
      );
      var focused := helpers.IntersectTokenSets(candidates, semanticContext);
      if |focused| > 0 {{
        helpers.BoostTokenLogits(lm, focused, 6.0);
      }}
    }}

    // Mask + choose directly so the boost survives. ConstrainedStep would
    // call GenerateLogits internally and overwrite the boost we just applied.
    lm.MaskValidNextAndEos(parser, currentConstrainedOut, eosToken);
    var next := lm.ChooseNextToken();
    helpers.cost := helpers.cost + 1;
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := steps;

// CSD_RATIONALE_BEGIN
// Boundary-rollback CSD with exclusion. Demonstrates RollbackAndExclude as
// a post-append recovery: after a constrained token is appended, the strategy
// checks the parser-valid continuation count; if it drops below a threshold,
// the strategy rewinds to the last boundary token and identifies the
// divergence token (the first token after the boundary that started the bad
// path). On the next iteration, the strategy penalizes that token to avoid
// repeating the same dead-end path.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Outside-span and close-span branches preserve the
//   invariant by their helper postconditions or by making the implication
//   vacuous. In the append-then-rollback branch, AppendConstrainedToken
//   gives parser.IsValidPrefix(currentConstrainedOut) immediately after the
//   append; if the rollback fires, RollbackAndExclude's ensures clause
//   directly gives parser.IsValidPrefix(rewound), and we assign
//   currentConstrainedOut := rewound.
// suffix: Append-only branches preserve the suffix by construction. In the
//   rollback branch, RollbackAndExclude's prefix-of ensures
//   (rewound == currentConstrainedOut[..|rewound|]) lets us drop exactly
//   |currentConstrainedOut| - |rewound| tokens off the end of `generated`,
//   restoring the equality between the suffix of generated and rewound.
// progress: Every iteration increments steps by exactly 1 and appends at
//   most one token to generated. Rollback only shrinks generated, never
//   grows it, so |generated| <= |generatedPrefix| + steps is preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var boundaryToken: Token := ",";
var narrowThreshold: int := 3;
var penaltyTokens: seq<Token> := [];

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
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
    if 0 < |penaltyTokens| && (forall t :: t in penaltyTokens ==> t in lm.Tokens) {{
      var next := helpers.PenalizedConstrainedStep(
        lm, parser, prompt + stablePrefix, currentConstrainedOut, penaltyTokens, 5.0, eosToken
      );
      penaltyTokens := [];
      steps := steps + 1;
      if next == eosToken {{
        break;
      }} else {{
        var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := appendedGenerated;
        insideConstrainedOut := appendedInside;
        currentConstrainedOut := appendedCurrent;
      }}
    }} else {{
      var g, i, c, hitEos := helpers.ConstrainedStep(lm, parser, prompt, generated, currentConstrainedOut, eosToken);
      steps := steps + 1;
      if hitEos {{
        break;
      }} else {{
        generated := g;
        insideConstrainedOut := i;
        currentConstrainedOut := c;

        var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
        if validCount < narrowThreshold && |currentConstrainedOut| > 0 {{
          var rewound, excludedTok, hasExcl := helpers.RollbackAndExclude(parser, currentConstrainedOut, boundaryToken);
          var dropped := |currentConstrainedOut| - |rewound|;
          generated := generated[..|generated| - dropped];
          currentConstrainedOut := rewound;
          if hasExcl && excludedTok in lm.Tokens {{
            penaltyTokens := [excludedTok];
          }}
        }}
      }}
    }}
  }}
}}

cost := steps;

// CSD_RATIONALE_BEGIN
// Group-aware vocabulary CSD. The runtime input validTokenGroups is a
// seq<seq<Token>> of caller-supplied token groups. Inside the constrained
// span, the strategy first flattens all groups into a single token bag and
// boosts grammar-valid candidates that fall in that bag. Independently, it
// also tracks an "active" group: when a sampled token belongs to a particular
// group, subsequent boosts inside the same span prefer that same group's
// tokens over the others. The threshold and group-activation policy are
// chosen locally by the strategy author.
//
// validTokenGroups is opaque to the strategy: no assumption is made about
// what makes a token "preferred" or what the groups represent. The strategy
// degrades to plain mask-and-choose when groups is empty or no overlap with
// the parser-valid candidate set exists.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: After MaskValidNextAndEos, any non-EOS token returned by
//   ChooseNextToken is parser-valid. Boosts only modify selection probability
//   on top of an already-masked logit vector; they cannot make masked tokens
//   selectable. AppendConstrainedToken preserves validity. CloseConstrainedSpan
//   makes the implication vacuous.
// suffix: AppendConstrainedToken and CloseConstrainedSpan maintain the suffix
//   invariant by their postconditions. Reading validTokenGroups is pure (no
//   state mutation) and does not affect generated or currentConstrainedOut.
// progress: Every non-break branch increments steps by 1 and appends at most
//   one token, preserving |generated| <= |generatedPrefix| + steps.
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
{{
  if !insideConstrainedOut {{
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      generated := generated + [next];
      if next == "<<" {{
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }}
    }}
  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    lm.GenerateLogits(constrainedPrompt + currentConstrainedOut);

    // Bias toward caller-preferred tokens when there is overlap with the
    // parser-valid candidate set. Use GroupHasValidMember as a cheap gate
    // before computing the boost.
    if |validTokenGroups| > 0 {{
      var flatPreferred := helpers.FlattenTokenGroups(validTokenGroups);
      if |flatPreferred| > 0 {{
        var anyValid := helpers.GroupHasValidMember(parser, currentConstrainedOut, flatPreferred);
        if anyValid {{
          var candidates := helpers.TopValidCandidates(
            lm, parser, constrainedPrompt, currentConstrainedOut, 30, eosToken
          );
          var preferred := helpers.IntersectTokenSets(candidates, flatPreferred);
          if |preferred| > 0 {{
            helpers.BoostTokenLogits(lm, preferred, 5.0);
          }}
        }}
      }}
    }}
    // Always apply the parser hard mask before sampling so AppendConstrained
    // Token's parser-validity precondition holds.
    lm.MaskValidNextAndEos(parser, currentConstrainedOut, eosToken);

    var next := lm.ChooseNextToken();
    helpers.cost := helpers.cost + 1;
    steps := steps + 1;
    if next == eosToken {{
      break;
    }} else {{
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }}
  }}
}}

cost := steps;
```

## Requirements
- Output a COMPLETE method body — no placeholder comments, no `/* YOUR CHOICE */`.
- The rationale block should briefly explain:
  - what state the strategy tracks
  - what conditions or observations it uses to choose its next action
  - how the tracked state is meant to support valid generation for this task
- Include the proof sketch block described in the system prompt. Dafny will
  check your proof for you — your sketch just has to convince a careful reader
  that each non-trivial loop invariant is preserved in every branch.
  This is a PROVABILITY discipline: novel strategies are fine as long as the
  proof story is coherent.
- If you call helpers with preconditions, track enough verifier-friendly state to make those preconditions provable at each call site.
- Make a real strategy choice; do not leave `...` or placeholders in the final output.
- Include a real decoding procedure, EOS handling, and state updates when needed by your design.
- Do not replace a failed attempt with a no-op or single-step fallback just to satisfy progress; keep a real multi-step decoding loop unless the task truly only needs one step.
"""

VERIFICATION_ERROR_REFINEMENT_PROMPT = """\
Your previous method body failed Dafny verification.

Task:
{task_description}

Previous attempt:
```dafny
{previous_strategy}
```

Verification error:
```
{error_message}
```
{structured_feedback_block}{error_history_block}{behavioral_context_block}
Your job is to make the MINIMUM edit that fixes this specific verification error
while preserving the rest of the strategy intact. The previous strategy's design
is yours to refine, not abandon.

Suggested approach:
1. Read the verification error and structured feedback carefully. Identify the
   single failing obligation (invariant, precondition, postcondition, or decreases)
   and the exact call site or line.
2. Add or strengthen only the invariants / guards / state updates needed to make
   that obligation provable.
3. Do NOT rewrite the loop body, change which helpers you call, or alter the
   overall strategy shape unless the error is fundamentally about that shape.
4. Keep the rationale block; update it only to note what you changed.

Mechanical fix hints:
- If the error is about `helpers.CloseConstrainedSpan(...)`, move that call into
  a branch already guarded by `parser.IsCompletePrefix(currentConstrainedOut)`
  instead of inferring completeness from a sampled token.
- If the error is about a logit-adjustment amount type, change integer literals
  like `5` or `100` to reals like `5.0` or `100.0`.
- If the error is about a missing invariant on `currentConstrainedOut`, add the
  invariant and re-establish it in each branch that modifies the state.

The harness will automatically restart you from scratch if your strategy is
structurally doomed across multiple attempts. Until that happens, preserve your
prior strategy and make focused fixes only.

Output ONLY a corrected full Dafny method body.
Do NOT output a method signature, outer wrapper text, or markdown fences.
Use only the contracts and tools already available in the system prompt.
"""

RUNTIME_ERROR_REFINEMENT_PROMPT = """\
Your method body passed Dafny verification but failed at runtime.

Previous attempt:
```dafny
{previous_strategy}
```

Runtime error:
```
{error_traceback}
```

Fix the runtime error. If needed, rewrite the method body instead of making only local edits.
Output ONLY a corrected method body (no signature, no braces, no markdown fences).
The corrected body must include the required rationale block at the top.
"""

COMPILATION_ERROR_REFINEMENT_PROMPT = """\
Your method body passed Dafny verification but failed during Dafny-to-Python compilation.

Previous attempt:
```dafny
{previous_strategy}
```

Compilation error:
```
{error_message}
```

Fix the compilation error. If needed, rewrite the method body instead of making only local edits.
Output ONLY a corrected method body (no signature, no braces, no markdown fences).
The corrected body must include the required rationale block at the top.
"""

FORMAT_REPAIR_PROMPT = """Your output must be a Dafny method body and is missing the required rationale block markers.

Rewrite the following content into a valid Dafny method body that preserves the same strategy semantics and outputs ONLY the method body.

Content to rewrite:
```dafny
{previous_strategy}
```
"""

EVALUATION_FAILURE_REFINEMENT_PROMPT = """\
Your method body passed verification and compilation but did not meet evaluation thresholds.

Task:
{task_description}

{best_strategy_block}
Previous attempt:
```dafny
{previous_strategy}
```

Evaluation results:
```
{evaluation_feedback}
```

{revision_guidance}

Output ONLY a corrected full Dafny method body.
Do NOT output a method signature, outer wrapper text, or markdown fences.
"""

def build_initial_prompt(task_description: str) -> tuple[str, str]:
    user_prompt = INITIAL_GENERATION_PROMPT.format(task_description=task_description)
    return SYSTEM_PROMPT, user_prompt

def build_verification_error_prompt(
    task_description: str,
    previous_strategy: str,
    error_message: str,
    behavioral_context: str = "",
    structured_feedback: str = "",
    error_history: str = "",
) -> tuple[str, str]:
    behavioral_context_block = ""
    structured_feedback_block = ""
    error_history_block = ""
    if behavioral_context:
        behavioral_context_block = (
            "\nRecent behavioral context from evaluation:\n```\n"
            f"{behavioral_context}\n"
            "```\n\n"
        )
    if structured_feedback:
        structured_feedback_block = (
            "\nStructured verifier analysis:\n```\n"
            f"{structured_feedback}\n"
            "```\n\n"
        )
    if error_history:
        error_history_block = (
            "\nRecent verification history across this run:\n```\n"
            f"{error_history}\n"
            "```\n\n"
        )
    user_prompt = VERIFICATION_ERROR_REFINEMENT_PROMPT.format(
        task_description=task_description,
        previous_strategy=previous_strategy,
        error_message=error_message,
        structured_feedback_block=structured_feedback_block,
        error_history_block=error_history_block,
        behavioral_context_block=behavioral_context_block,
    )
    return SYSTEM_PROMPT, user_prompt

def build_runtime_error_prompt(previous_strategy: str, error_traceback: str) -> tuple[str, str]:
    user_prompt = RUNTIME_ERROR_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        error_traceback=error_traceback,
    )
    return SYSTEM_PROMPT, user_prompt

def build_compilation_error_prompt(previous_strategy: str, error_message: str) -> tuple[str, str]:
    user_prompt = COMPILATION_ERROR_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        error_message=error_message,
    )
    return SYSTEM_PROMPT, user_prompt

def build_format_repair_prompt(previous_strategy: str) -> tuple[str, str]:
    user_prompt = FORMAT_REPAIR_PROMPT.format(previous_strategy=previous_strategy)
    return SYSTEM_PROMPT, user_prompt

def build_evaluation_failure_prompt(
    task_description: str,
    previous_strategy: str,
    evaluation_feedback: str,
    best_strategy: str = "",
    best_accuracy: float = 0.0,
    current_accuracy: float = 0.0,
    min_accuracy: float = 0.75,
) -> tuple[str, str]:
    # Decide revision mode based on how far current accuracy is from target
    gap = min_accuracy - current_accuracy

    if gap > 0.40:
        # Very far (e.g. 7% vs 75%) — the strategy is fundamentally broken
        revision_guidance = (
            "The current strategy is fundamentally broken (accuracy far below target). "
            "You should redesign the approach. Study the failure modes carefully and "
            "write a new strategy that addresses the root cause. Do not make small "
            "tweaks to a broken foundation — rethink the overall structure.\n"
            "Before choosing a new direction, check the 'Approaches that caused regressions' "
            "list at the end of the evaluation history. If an approach or idea already "
            "regressed in prior attempts, do NOT try another variant of it — choose a "
            "fundamentally different technique.\n"
            "The revised rationale should explain your new approach."
        )
    elif gap > 0.15:
        # Moderately far (e.g. 50% vs 75%) — significant changes needed but
        # preserve what works
        revision_guidance = (
            "The strategy has the right general shape but significant issues remain. "
            "You may make a few targeted changes, but study the failure modes to "
            "understand what is working and what is not. Preserve the parts that work "
            "and fix the parts that don't.\n"
            "Before proposing changes, check the 'Approaches that caused regressions' "
            "list at the end of the evaluation history. If previous attempts already "
            "targeted the same failure mode with a similar approach and regressed, do "
            "not try another variant — choose a different failure mode or a fundamentally "
            "different technique.\n"
            "The revised rationale should explain what you changed and why."
        )
    else:
        # Close (e.g. 70% vs 75%) — one change at a time
        revision_guidance = (
            "IMPORTANT — you are close to the target. Make exactly ONE change at a time:\n"
            "1. Look at the failure modes above. Identify the single largest failure category.\n"
            "2. Devise the smallest possible code edit that addresses that one failure mode.\n"
            "3. Copy the best-performing strategy verbatim and apply only that one edit. "
            "Do not rename variables, reorganize code, swap helpers, adjust unrelated "
            "parameters, or 'clean up' anything that is already working.\n"
            "4. If your one change requires a new helper, that is fine — but do not "
            "simultaneously change other parts of the strategy.\n\n"
            "Strategies that make multiple simultaneous changes tend to regress because "
            "untested interactions cancel out the intended fix. One change, test, iterate.\n"
            "Before proposing your change, check the 'Approaches that caused regressions' "
            "list at the end of the evaluation history. If previous attempts already "
            "targeted the same failure mode with a similar approach and regressed, do "
            "not try another variant of it — choose a different failure mode or a "
            "fundamentally different technique.\n"
            "The revised rationale should name the single failure mode targeted and the single edit made."
        )

    best_strategy_block = ""
    if best_strategy:
        if best_strategy == previous_strategy:
            best_strategy_block = (
                f"\nThe previous attempt IS the best-performing strategy so far "
                f"({best_accuracy:.1%} accuracy).\n\n"
            )
        else:
            best_strategy_block = (
                f"\nBest-performing strategy so far ({best_accuracy:.1%} accuracy) — "
                f"use this as your reference for what works:\n"
                f"```dafny\n{best_strategy}\n```\n\n"
            )
    user_prompt = EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
        task_description=task_description,
        previous_strategy=previous_strategy,
        evaluation_feedback=evaluation_feedback,
        best_strategy_block=best_strategy_block,
        revision_guidance=revision_guidance,
    )
    return SYSTEM_PROMPT, user_prompt
