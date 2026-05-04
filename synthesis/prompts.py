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
    requires eosToken in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    ensures cost <= maxSteps
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained
    decreases maxSteps

## Output rules
- Start with: `// CSD_RATIONALE_BEGIN\n// ...\n// CSD_RATIONALE_END`
- Immediately after the rationale, emit a proof sketch block:
  `// CSD_PROOF_SKETCH_BEGIN\n// ...\n// CSD_PROOF_SKETCH_END`.
- Initialize all out-parameters before any loop/recursion.
- Assign `cost` before returning.
- Do NOT redeclare out-parameters as locals.
- Use the provided `helpers` instance (type `CSDHelpers`). Do NOT write `var helpers := new CSDHelpers();`.
- Do NOT use `CSDHelpers.<Method>` for instance methods.

## API Guidance

- `Token` is type `string`.
- `prompt` is type `Prefix` (= `seq<Token>`).
- `generated` / `generatedPrefix` contain the full answer text, including delimiter tokens.
- `currentConstrained` / `currentConstrainedOut` track only the active constrained segment contents between delimiters.
- EOS is terminal.

## Available Tools

### Runtime inputs
```
validTokenGroups: seq<seq<Token>>
```
`validTokenGroups` is caller-supplied contextual vocabulary. It may be empty,
inner groups may be empty, groups may overlap, and groups may contain tokens
outside `lm.Tokens`; safe helpers internally ignore non-vocabulary tokens.

### Helper methods
```
var next := helpers.UnconstrainedStep(lm, prompt, generated);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.OpenConstrainedSpan(lm, generated);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.EnterObservedConstrainedSpan(lm, generated);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrained, next);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrained);
var next := helpers.ConstrainedStep(lm, parser, prompt, currentConstrained, eosToken);
var next, wasConstrained := helpers.ConfidenceGatedStep(lm, parser, prompt, currentConstrained, eosToken);
helpers.SafeBoostTokenLogits(lm, tokens, amount);
helpers.SafePenalizeTokenLogits(lm, tokens, amount);
var next := helpers.SafeBoostedConstrainedStep(lm, parser, prompt, currentConstrained, tokensToBoost, 4.0, eosToken);
var next := helpers.SafePenalizedConstrainedStep(lm, parser, prompt, currentConstrained, tokensToPenalize, 4.0, eosToken);
var next := helpers.SafeRepetitionPenaltyStep(lm, parser, prompt, currentConstrained, generated, 2.0, eosToken);
var next := helpers.SafeTemperatureConstrainedStep(lm, parser, prompt, currentConstrained, 0.8, eosToken);
var next, usedFallback := helpers.SafeSoftConstrainedStep(lm, parser, prompt, currentConstrained, 8.0, eosToken);
var next := helpers.GroupBoostedConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, 4.0, eosToken);
var next := helpers.AdaptiveConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, 4.0, 12, eosToken);
var generatedOut, stoppedOnOpenSpan, stoppedOnEos, stepsUsed := helpers.UnconstrainedChunk(lm, prompt, generated, maxChunkTokens, openSpanToken, eosToken);
var currentOut, hitEos, stepsUsed := helpers.ConstrainedSymbol(lm, parser, constrainedPrompt, currentConstrained, stepTokenBudget, eosToken);
var generatedOut, currentOut, hitEos, stepsUsed := helpers.ConstrainedSymbolInGenerated(lm, parser, constrainedPrompt, generated, currentConstrained, stepTokenBudget, eosToken);
```
`OpenConstrainedSpan` appends a new `"<<"` token and costs 1 step. If `"<<"`
was already emitted by `UnconstrainedStep` or `UnconstrainedChunk`, use
`EnterObservedConstrainedSpan` to update span state without appending another
delimiter or consuming additional token budget.

### Parser queries
```
var narrow := helpers.DeadEndDetection(parser, currentConstrained, minValidCount);
var count := helpers.ValidTokenCount(parser, currentConstrained);
var valid := helpers.IsTokenValidNext(parser, currentConstrained, token);
var candidates := helpers.TopValidCandidates(lm, parser, prompt, currentConstrained, maxCandidates, eosToken);
var rolled := helpers.RollbackToValidPrefix(parser, constrainedPrefix);
var generatedOut, currentOut := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrained);
var flat := helpers.FlattenTokenGroups(validTokenGroups);
var groupIdx := helpers.GroupContaining(validTokenGroups, token);
var prevTok, foundPrev := helpers.LastTokenBefore(generated, ">>");
var following := helpers.ExtractAfterKeyword(prefix, keyword);
var intersection := helpers.IntersectTokenSets(a, b);
var difference := helpers.SubtractTokenSets(a, b);
```

## Tool API Reference

Use this as API documentation for what each helper accomplishes in a constrained
decoding strategy. These descriptions define library behavior, not task-specific
revision rules.

`cost` is token-step budget use, not wall-clock time and not a count of helper
calls. LM-generated tokens count even if the token is EOS or later rejected from
visible output. A delimiter token appended by a helper counts as one token-step.
Bookkeeping, parser queries, token-set transforms, and logit edits do not
consume token budget by themselves.

### Outside-span generation

- `helpers.UnconstrainedStep(lm, prompt, generated)`
  Role: one free LM token outside parser control.
  Mechanics: calls the LM on `prompt + generated` and returns one token; the
  strategy appends non-EOS tokens itself.
  Cost: +1 token-step, including EOS.
  Control profile: maximum free-LM continuation, no parser control.

- `helpers.UnconstrainedChunk(lm, prompt, generated, maxChunkTokens, "<<", eosToken)`
  Role: a short free-LM continuation that can stop when an opening delimiter is
  naturally emitted.
  Mechanics: returns `generatedOut` already extended by the non-EOS chunk. If
  `stoppedOnOpenSpan` is true, `generatedOut` already ends with `"<<"`.
  Cost: +`stepsUsed`; EOS counts in `stepsUsed` but is not appended.
  Control profile: free-LM continuation with delimiter observation.

- `helpers.OpenConstrainedSpan(lm, generated)`
  Role: explicit transition from free generation into constrained generation.
  Mechanics: appends visible `"<<"`, sets `insideOut := true`, and resets
  `currentOut := []`.
  Cost: +1 token-step for the forced delimiter.
  Control profile: direct delimiter/state control, no LM sampling.

- `helpers.EnterObservedConstrainedSpan(lm, generated)`
  Role: state transition after `"<<"` is already present in visible output.
  Mechanics: leaves `generated` unchanged, sets `insideOut := true`, and resets
  `currentOut := []`.
  Cost: +0.
  Control profile: bookkeeping only.

### Inside-span generation and state updates

- `helpers.ConstrainedStep(lm, parser, prompt, currentConstrained, eosToken)`
  Role: one parser-valid token choice inside a constrained span.
  Mechanics: calls the LM, hard-masks to parser-valid next tokens plus EOS, and
  returns one token without appending it.
  Cost: +1 token-step, including EOS.
  Control profile: strongest token-level parser control.

- `helpers.ConfidenceGatedStep(lm, parser, prompt, currentConstrained, eosToken)`
  Role: one inside-span token choice that uses hard parser control only when
  the LM's current top token would not preserve parser validity.
  Mechanics: calls the LM, reads the highest-logit token, returns it directly
  if it is EOS or keeps `currentConstrained + [token]` parser-valid; otherwise
  hard-masks to parser-valid next tokens plus EOS and samples from that mask.
  Returns `wasConstrained == true` only on the hard-mask fallback path.
  Cost: +1 token-step, including EOS.
  Control profile: LM-preferred continuation when already parser-compatible,
  hard parser fallback otherwise.

- `helpers.AppendConstrainedToken(lm, parser, generated, currentConstrained, next)`
  Role: commit a previously selected parser-valid token into visible and
  constrained state.
  Mechanics: appends `next` to both `generated` and `currentConstrained`.
  Cost: +0; the token was already counted by the generation helper.
  Control profile: state synchronization only.

- `helpers.ConstrainedSymbol(lm, parser, constrainedPrompt, currentConstrained, maxSymbolTokens, eosToken)`
  Role: multi-token constrained progress from one LM chunk.
  Mechanics: generates up to `maxSymbolTokens`, then accepts only the longest
  parser-valid prefix of that chunk.
  Cost: +`stepsUsed`; rejected suffix tokens and EOS still count.
  Control profile: chunk-level LM continuation with parser-prefix acceptance.

- `helpers.ConstrainedSymbolInGenerated(lm, parser, constrainedPrompt, generated, currentConstrained, maxSymbolTokens, eosToken)`
  Role: multi-token constrained progress while updating full visible output.
  Mechanics: computes the stable prefix, calls `ConstrainedSymbol`, then returns
  `generatedOut := stablePrefix + currentOut`.
  Cost: +`stepsUsed`, which may exceed visible output growth.
  Control profile: chunk-level LM continuation plus generated/current state
  reconstruction.

- `helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrained)`
  Role: exit a complete constrained span.
  Mechanics: requires `parser.IsCompletePrefix(currentConstrained)`, appends
  visible `">>"` unless already emitted by the constrained grammar, exits
  constrained mode, and clears `currentOut`.
  Cost: +1 token-step for the close action.
  Control profile: direct delimiter/state control gated by parser completeness.

### Soft preferences and group-aware constrained decoding

- `helpers.GroupBoostedConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, amount, eosToken)`
  Role: one parser-valid token choice with caller-supplied group preferences.
  Mechanics: calls the LM, boosts groups containing parser-valid members, then
  hard-masks to parser-valid next tokens plus EOS.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with soft preference among legal choices.

- `helpers.AdaptiveConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, amount, narrowThreshold, eosToken)`
  Role: one parser-valid token choice with group preferences applied only at
  narrower parser states.
  Mechanics: same hard mask as `ConstrainedStep`; group boosts are applied only
  when `parser.ValidNextTokenCount(currentConstrained) <= narrowThreshold`.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with conditional soft preference.

- `helpers.SafeBoostedConstrainedStep(lm, parser, prompt, currentConstrained, tokens, amount, eosToken)`
  Role: one parser-valid token choice with a soft boost for a caller-supplied
  token set.
  Mechanics: calls the LM, ignores any `tokens` outside `lm.Tokens`, boosts the
  remaining listed tokens, then hard-masks to parser-valid next tokens plus EOS.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with token-list soft preference.

- `helpers.SafePenalizedConstrainedStep(lm, parser, prompt, currentConstrained, tokens, amount, eosToken)`
  Role: one parser-valid token choice with a soft penalty for a caller-supplied
  token set.
  Mechanics: calls the LM, ignores any `tokens` outside `lm.Tokens`, penalizes
  the remaining listed tokens, then hard-masks to parser-valid next tokens plus
  EOS.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with token-list soft avoidance.

- `helpers.SafeRepetitionPenaltyStep(lm, parser, prompt, currentConstrained, generated, amount, eosToken)`
  Role: one parser-valid token choice that discourages reusing tokens already
  present in the visible output.
  Mechanics: calls the LM, filters `generated` through `lm.Tokens`, penalizes
  those tokens, then hard-masks to parser-valid next tokens plus EOS.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with repetition avoidance.

- `helpers.SafeTemperatureConstrainedStep(lm, parser, prompt, currentConstrained, temperature, eosToken)`
  Role: one parser-valid token choice with local sampling sharpness adjusted.
  Mechanics: calls the LM, clamps `temperature` to a safe range, scales logits,
  then hard-masks to parser-valid next tokens plus EOS.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with sharper or flatter token sampling.

- `helpers.SafeSoftConstrainedStep(lm, parser, prompt, currentConstrained, boostAmount, eosToken)`
  Role: one inside-span token choice that first tries a soft grammar preference,
  then falls back to hard parser control if the soft choice would not preserve
  parser validity.
  Mechanics: calls the LM, boosts parser-valid next tokens plus EOS, samples one
  soft choice, and returns it if it preserves parser validity; otherwise it
  hard-masks to parser-valid next tokens plus EOS and returns the fallback token.
  Returns `usedFallback == true` only when the hard fallback path is used.
  Cost: +1 token-step by helper contract, including EOS.
  Control profile: soft grammar preference with hard parser fallback.

- `helpers.SafeBoostTokenLogits(lm, tokens, amount)` and
  `helpers.SafePenalizeTokenLogits(lm, tokens, amount)`
  Role: adjust soft preference in the current logits.
  Mechanics: filter `tokens` through `lm.Tokens`, then add to or subtract from
  their existing logits. They do not call the LM, sample, append output, or
  inspect the parser.
  Cost: +0.
  Control profile: soft logit preference only; relevant only to later choices
  that read the modified logits rather than regenerating fresh logits first.

### Parser queries, repair, and context extraction

- `helpers.ValidTokenCount(parser, currentConstrained)` and
  `helpers.DeadEndDetection(parser, currentConstrained, minValidCount)`
  Role: inspect parser branching at the current constrained prefix.
  Mechanics: return a count or thresholded boolean; no LM call and no state
  change.
  Cost: +0.
  Control profile: parser information only.

- `helpers.IsTokenValidNext(parser, currentConstrained, token)`
  Role: test one candidate token against the parser.
  Mechanics: returns whether `token` is valid next; no state change.
  Cost: +0.
  Control profile: parser information only.

- `helpers.TopValidCandidates(lm, parser, prompt, currentConstrained, maxCandidates, eosToken)`
  Role: inspect the LM's ranking among legal next tokens.
  Mechanics: calls the LM once and returns up to `maxCandidates` high-logit
  parser-valid candidates, with EOS admissible. It does not append a candidate.
  Cost: +1 token-step.
  Control profile: LM-ranked parser-valid candidate information.

- `helpers.RollbackConstrainedSuffix(parser, generated, currentConstrained)`
  Role: repair active constrained state by shortening the constrained suffix.
  Mechanics: computes the stable prefix from the current suffix length, rolls
  back only `currentConstrained` until parser-valid, and reconstructs
  `generatedOut`.
  Cost: +0.
  Control profile: parser repair by deletion.

- `helpers.LastTokenBefore(generated, sep)` and
  `helpers.ExtractAfterKeyword(prefix, keyword)`
  Role: read lightweight context from existing generated tokens.
  Mechanics: scan token sequences and return matching context tokens; no LM
  call and no state change.
  Cost: +0.
  Control profile: context information only.

- `helpers.FlattenTokenGroups`, `helpers.GroupContaining`,
  `helpers.IntersectTokenSets`, and `helpers.SubtractTokenSets`
  Role: transform token sets or groups.
  Mechanics: operate on sequences only; no LM call, parser query, output append,
  or state transition.
  Cost: +0.
  Control profile: token-set bookkeeping only.

## Proof sketch discipline

Your output must include a `// CSD_PROOF_SKETCH_BEGIN ... // CSD_PROOF_SKETCH_END`
block between the rationale and the method body.

For each of the following two non-trivial loop invariants, explain in one or
two sentences per branch why that branch preserves the invariant:

1. `parser_validity`:
   `insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)`
2. `progress`:
   `|generated| <= |generatedPrefix| + steps`
   Each branch should increment `steps` by the token budget consumed by its
   generation/forced-delimiter helper. Some helpers may consume tokens that are
   not appended visibly, for example EOS or a rejected suffix from
   `ConstrainedSymbol`; in those cases, the output-length bound is still
   preserved because visible growth is at most the consumed token budget.

The proof sketch should explain preservation of the listed invariants.

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
//   ConstrainedStep branch we only call AppendConstrainedToken when
//   IsTokenValidNext holds, so the appended prefix remains valid.
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
  invariant |generated| <= |generatedPrefix| + steps
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
    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
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
// Context-triggered CSD. The strategy tracks whether a neutral local marker has
// recently appeared in the free text. When that marker is seen, the next
// outside-span action opens a constrained span before returning to ordinary
// delimiter-triggered behavior. Inside the span, parser validity remains the
// hard authority.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Outside the span, the implication is vacuous. The
//   context-triggered open branch uses OpenConstrainedSpan, which returns
//   currentConstrainedOut == [], valid by parser.IsValidPrefix([]). If the
//   free token itself is "<<", we enter with currentConstrainedOut := [].
//   CloseConstrainedSpan exits constrained mode. ConstrainedStep plus
//   AppendConstrainedToken preserves validity for non-EOS tokens.
// progress: UnconstrainedStep, OpenConstrainedSpan, CloseConstrainedSpan, and
//   ConstrainedStep each consume one step and append at most one token, so
//   |generated| <= |generatedPrefix| + steps is preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var markerArmed: bool := false;
var markerToken: Token := ":";

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{{
  if !insideConstrainedOut {{
    if markerArmed {{
      var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);
      generated := openedGenerated;
      insideConstrainedOut := openedInside;
      currentConstrainedOut := openedCurrent;
      markerArmed := false;
      steps := steps + 1;
    }} else {{
      var next := helpers.UnconstrainedStep(lm, prompt, generated);
      steps := steps + 1;
      if next == eosToken {{
        break;
      }} else {{
        generated := generated + [next];
        if next == "<<" {{
          insideConstrainedOut := true;
          currentConstrainedOut := [];
          markerArmed := false;
        }} else if next == markerToken {{
          markerArmed := true;
        }}
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
    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
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
// Group-aware constrained CSD. Generate freely until "<<" appears, then use
// caller-supplied token groups as a soft preference while the parser remains
// the hard validity authority.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Outside the span, the implication is vacuous unless next is
//   "<<", in which case currentConstrainedOut := [] is valid by precondition.
//   CloseConstrainedSpan flips insideConstrainedOut to false. In the active
//   constrained branch, GroupBoostedConstrainedStep returns either EOS or a
//   parser-valid next token, and AppendConstrainedToken preserves validity.
// progress: Every branch appends at most one token and steps grows by 1, so
//   |generated| <= |generatedPrefix| + steps is preserved.
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
  invariant |generated| <= |generatedPrefix| + steps
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
    var next := helpers.GroupBoostedConstrainedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, eosToken
    );
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
// Top-candidate constrained CSD. Inside a span, query a small ranked set of
// parser-valid candidates and append the first non-EOS candidate if available.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Enter only via the unconstrained branch when next == "<<",
//   at which point currentConstrainedOut := [] (valid). Complete-prefix branch
//   flips insideConstrainedOut to false via CloseConstrainedSpan. TopValidCandidates
//   returns only EOS or valid-next tokens; after excluding EOS, AppendConstrainedToken
//   preserves validity.
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
  invariant |generated| <= |generatedPrefix| + attempts
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
    var next := candidates[0];
    if next == eosToken && |candidates| > 1 {{
      next := candidates[1];
    }}
    attempts := attempts + 1;
    if next == eosToken {{
      break;
    }} else {{
      assert next in candidates;
      assert next in parser.ValidNextTokens(currentConstrainedOut);
      assert parser.IsValidPrefix(currentConstrainedOut + [next]);
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
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
// cost (ensures cost <= maxSteps): each non-base path consumes one local step
//   before recursing with maxSteps - 1. The returned cost is 1 plus the
//   recursive sub-cost, so induction on maxSteps gives cost <= maxSteps.
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
    cost := 1;
  }} else {{
    var nextGenerated := generatedPrefix + [next];
    if next == "<<" {{
      var subGenerated, subInside, subCurrent, subCost := MyCSDStrategy(
        lm, parser, prompt, nextGenerated, true, [], maxSteps - 1, stepTokenBudget, validTokenGroups, eosToken
      );
      generated := subGenerated;
      insideConstrainedOut := subInside;
      currentConstrainedOut := subCurrent;
      cost := 1 + subCost;
    }} else {{
      var subGenerated, subInside, subCurrent, subCost := MyCSDStrategy(
        lm, parser, prompt, nextGenerated, false, [], maxSteps - 1, stepTokenBudget, validTokenGroups, eosToken
      );
      generated := subGenerated;
      insideConstrainedOut := subInside;
      currentConstrainedOut := subCurrent;
      cost := 1 + subCost;
    }}
  }}
}} else if parser.IsCompletePrefix(currentConstrained) {{
  var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
    lm, parser, generatedPrefix, currentConstrained
  );
  var subGenerated, subInside, subCurrent, subCost := MyCSDStrategy(
    lm, parser, prompt, closedGenerated, closedInside, closedCurrent, maxSteps - 1, stepTokenBudget, validTokenGroups, eosToken
  );
  generated := subGenerated;
  insideConstrainedOut := subInside;
  currentConstrainedOut := subCurrent;
  cost := 1 + subCost;
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
    cost := 1;
  }} else {{
    assert chosen in candidates;
    assert chosen in parser.ValidNextTokens(currentConstrained);
    assert parser.IsValidPrefix(currentConstrained + [chosen]);
    var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
      lm, parser, generatedPrefix, currentConstrained, chosen
    );
    var subGenerated, subInside, subCurrent, subCost := MyCSDStrategy(
      lm, parser, prompt, appendedGenerated, appendedInside, appendedCurrent, maxSteps - 1, stepTokenBudget, validTokenGroups, eosToken
    );
    generated := subGenerated;
    insideConstrainedOut := subInside;
    currentConstrainedOut := subCurrent;
    cost := 1 + subCost;
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
  invariant |generated| <= |generatedPrefix| + steps
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
    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
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
// Symbol-step CSD. Each outer loop iteration is one "symbol step" — the model
// is called for up to stepTokenBudget tokens at once, and the longest valid
// parser prefix of the result is accepted. This aligns generation granularity
// with the task's natural units: SQL keywords, arithmetic expressions, or
// multi-subword identifiers can be emitted as a unit instead of being forced
// token by token. Outside a constrained span the strategy generates one token
// freely. Inside the span it uses ConstrainedSymbol, passing stepTokenBudget
// as the per-step token allowance. Close as soon as the parser reports the
// prefix is complete.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Opening a span sets currentConstrainedOut := [], valid by
//   precondition. CloseConstrainedSpan flips insideConstrainedOut to false.
//   ConstrainedSymbol postcondition: parser.IsValidPrefix(symbolOut).
// progress: steps advances by exactly 1 for UnconstrainedStep and
//   CloseConstrainedSpan (each appends ≤1 token), and by stepsUsed ≥ 1 for
//   ConstrainedSymbol (postcondition). So decreases maxSteps - steps always
//   decreases. The invariant |generated| <= |generatedPrefix| + steps is
//   linear: each branch adds at most the consumed token budget to visible
//   output and advances steps by that budget (or 1 for single-token branches). ✓
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
  invariant |generated| <= |generatedPrefix| + steps
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
    var constrainedPrompt := prompt + stablePrefix;
    var symbolBudget: nat := maxSteps - steps;
    var symbolGenerated, symbolOut, hitEos, stepsUsed := helpers.ConstrainedSymbolInGenerated(
      lm, parser, constrainedPrompt, generated, currentConstrainedOut, symbolBudget, eosToken
    );
    generated := symbolGenerated;
    currentConstrainedOut := symbolOut;
    steps := steps + stepsUsed;
    if hitEos {{
      break;
    }}
  }}
}}

cost := steps;
```

```dafny
// CSD_RATIONALE_BEGIN
// Adaptive-narrowness CSD. Inside a constrained span, query the parser's
// valid-continuation count and choose either one-token ConstrainedStep or a
// bounded ConstrainedSymbol call. The `narrowThreshold` local controls which
// branch is used.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: CloseConstrainedSpan makes implication vacuous. In the
//   tight branch, ConstrainedStep returns a parser-valid next token and
//   AppendConstrainedToken preserves validity. In the loose branch,
//   ConstrainedSymbol postcondition guarantees parser.IsValidPrefix(symbolOut).
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
  invariant |generated| <= |generatedPrefix| + steps
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
      var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
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
      var remaining: nat := maxSteps - steps;
      var symbolBudget: nat := if stepTokenBudget == 0 || stepTokenBudget > remaining then remaining else stepTokenBudget;
      var symbolGenerated, symbolOut, hitEos, stepsUsed := helpers.ConstrainedSymbolInGenerated(
        lm, parser, constrainedPrompt, generated, currentConstrainedOut, symbolBudget, eosToken
      );
      generated := symbolGenerated;
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
// Context-tracking CSD. Maintains a strategy-local seq<Token> across loop
// iterations. After each constrained token append, it queries the span for
// tokens following a keyword via ExtractAfterKeyword. At candidate-selection
// positions, it intersects parser-valid candidates with that context set and
// boosts the intersection.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: GroupBoostedConstrainedStep returns either EOS or a
//   parser-valid next token; the EOS branch breaks. AppendConstrainedToken
//   preserves validity. CloseConstrainedSpan makes the implication vacuous.
//   The context variable is never passed to the parser as authority, so it
//   cannot affect parser_validity.
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
  invariant |generated| <= |generatedPrefix| + steps
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
    var groups := validTokenGroups;
    if |semanticContext| > 0 {{
      groups := [semanticContext] + validTokenGroups;
    }}
    var next := helpers.GroupBoostedConstrainedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut, groups, 6.0, eosToken
    );
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
// Rollback-on-stall CSD. Generate freely until "<<" appears. Inside the span,
// generate parser-valid tokens, but if the active constrained content grows
// beyond a local rollback limit before becoming complete, roll back only the
// constrained suffix to the nearest valid non-dead prefix and continue from
// that repaired state.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Opening a span sets currentConstrainedOut := [], which is
//   valid. Closing a complete span exits constrained mode. ConstrainedStep plus
//   AppendConstrainedToken preserves validity on non-EOS tokens. The rollback
//   branch uses RollbackConstrainedSuffix, whose postcondition gives a valid
//   repaired currentConstrainedOut.
// progress: UnconstrainedStep, CloseConstrainedSpan, and ConstrainedStep each
//   consume one step and append at most one token. RollbackConstrainedSuffix
//   shrinks or preserves generated and we still increment steps by 1, so the
//   output-length bound remains true while the loop metric decreases.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var rollbackLimit: nat := 24;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
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
  }} else if |currentConstrainedOut| >= rollbackLimit {{
    var rolledGenerated, rolledCurrent := helpers.RollbackConstrainedSuffix(
      parser, generated, currentConstrainedOut
    );
    generated := rolledGenerated;
    insideConstrainedOut := true;
    currentConstrainedOut := rolledCurrent;
    steps := steps + 1;
  }} else {{
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
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
// Logit-shaped constrained CSD. The strategy uses ordinary free generation
// outside spans. Inside a span, parser validity remains hard; the only extra
// policy is a local soft preference: avoid closing very short constrained
// prefixes and softly prefer operator tokens once the prefix has begun. The
// safe helpers filter literal token lists internally, so the strategy does not
// need separate vocabulary-membership state for those lists.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Outside the span the implication is vacuous unless a "<<"
//   token is observed, in which case currentConstrainedOut becomes [].
//   CloseConstrainedSpan exits constrained mode. Both safe constrained-step
//   helpers return EOS or a token preserving parser validity, and
//   AppendConstrainedToken carries that validity into currentConstrainedOut.
// progress: UnconstrainedStep, CloseConstrainedSpan, and each safe constrained
//   step consume one token-step and append at most one visible token, so
//   |generated| <= |generatedPrefix| + steps is preserved.
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
  invariant |generated| <= |generatedPrefix| + steps
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
    var next := eosToken;
    if |currentConstrainedOut| < 2 {{
      next := helpers.SafePenalizedConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, [">>"], 6.0, eosToken
      );
    }} else {{
      next := helpers.SafeBoostedConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, ["+", "-", "*", "/"], 2.0, eosToken
      );
    }}
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
- If you call helpers with preconditions, track enough verifier-friendly state to make those preconditions provable at each call site.
- Include a real decoding procedure, EOS handling, and state updates when needed by your design.
"""


_VERIFIED_EXAMPLES_HEADER = "## Verified Examples\n\n"
_REQUIREMENTS_HEADER = "\n\n## Requirements"
VERIFIED_EXAMPLES = INITIAL_GENERATION_PROMPT.split(_VERIFIED_EXAMPLES_HEADER, 1)[1].split(
    _REQUIREMENTS_HEADER,
    1,
)[0]
INITIAL_GENERATION_PROMPT = INITIAL_GENERATION_PROMPT.replace(
    VERIFIED_EXAMPLES,
    "{verified_examples}",
    1,
)


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
Revise the method body so it verifies.

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
Your method body passed verification and compilation, then was evaluated on the task,
but did not meet evaluation thresholds.
All method parameters in the Dafny signature are available to the strategy.
Treat the evaluation results below as factual observations of generated outputs.

Task:
{task_description}

## Strategy Context

Previous/current evaluated attempt:
```dafny
{previous_strategy}
```
{working_hypothesis_block}

Evaluation results:
```
{evaluation_feedback}
```
{evaluation_history_block}

## Verified Examples

These are verified reference CSD patterns available for reuse during refinement.
Use them as examples of valid helper usage, state tracking, loop structure, and
proof style. The evaluation results and recent history above determine which
parts, if any, are relevant to the next strategy revision.

{verified_examples}

Recent evaluation history is provided for context.
Use the evaluation history to avoid repeating strategies that already failed.
Prefer substantive changes over small parameter-only tweaks when the prior
strategy structure performed poorly. Preserve and improve ideas that were
closer to the thresholds.
Best-so-far means the strategy with the best balanced progress on both accuracy
and syntax. A strategy that is strong on only one metric but weak on the other
is not best-so-far merely because one score is high.
Avoid small parameter tweaks to a strategy shape that repeatedly underperformed.
If a shape regressed multiple times, make a structurally different change.
If a shape is best-so-far, preserve its core unless changing a clearly isolated
failure mode.
Output ONLY a corrected full Dafny method body.
Do NOT output a method signature, outer wrapper text, or markdown fences.
The revised rationale should explain what changed in response to the evaluation results.
"""


def build_initial_prompt(task_description: str) -> tuple[str, str]:
    user_prompt = INITIAL_GENERATION_PROMPT.format(
        task_description=task_description,
        verified_examples=VERIFIED_EXAMPLES,
    )
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
    evaluation_history: str = "",
    working_hypothesis: str = "",
) -> tuple[str, str]:
    evaluation_history_block = ""
    working_hypothesis_block = ""
    if evaluation_history:
        evaluation_history_block = (
            "\nRecent evaluation history:\n```\n"
            f"{evaluation_history}\n"
            "```\n"
        )
    if working_hypothesis:
        working_hypothesis_block = (
            "\n\n"
            f"{working_hypothesis}\n"
        )
    user_prompt = EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
        task_description=task_description,
        previous_strategy=previous_strategy,
        working_hypothesis_block=working_hypothesis_block,
        evaluation_feedback=evaluation_feedback,
        evaluation_history_block=evaluation_history_block,
        verified_examples=VERIFIED_EXAMPLES,
    )
    return SYSTEM_PROMPT, user_prompt
