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

import re

try:
    from synthesis.prompt_rendering import render as _render_prompt
    from synthesis.prompt_rendering.models.author_prompts import (
        CompilationErrorPromptModel,
        EvaluationFailurePromptModel,
        FormatRepairPromptModel,
        InitialPromptModel,
        RuntimeErrorPromptModel,
        VerificationErrorPromptModel,
    )
except ImportError:
    from prompt_rendering import render as _render_prompt
    from prompt_rendering.models.author_prompts import (
        CompilationErrorPromptModel,
        EvaluationFailurePromptModel,
        FormatRepairPromptModel,
        InitialPromptModel,
        RuntimeErrorPromptModel,
        VerificationErrorPromptModel,
    )

# NOTE:
# The synthesized output is injected into
# `synthesis/verify/library/GeneratedCSD.dfy` as the BODY
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
- Assign the existing out-parameters directly; they are already in scope.
- Use the provided `helpers` instance (type `CSDHelpers`).
- Call static entries listed as `CSDHelpers.<Method>` with that qualifier.
- Call instance helper methods as `helpers.<Method>`.

## API Guidance

- `Token` is type `string`.
- `prompt` is type `Prefix` (= `seq<Token>`).
- `generated` / `generatedPrefix` contain the full answer text, including delimiter tokens.
- `currentConstrained` / `currentConstrainedOut` track only the active constrained segment contents between delimiters.
- EOS is terminal.
- Visible delimiters such as `"<<"` and `">>"` are task-contract artifacts.
  Use visible delimiters only when the task or evaluator requires visible constrained spans.
  For hidden constrained chunks, fully constrained objects, or another structured-output surface,
  emit the task-native surface.
- Never detect a visible delimiter with exact token equality such as
  `next == "<<"` or `next == ">>"`. Tokenizers may attach whitespace or split a
  delimiter across tokens. Append the token first, then test the full relevant
  prefix with `RenderedEndsWith(generated, "<<")` or the matching suffix.

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
helpers.AppendTaskGuidance(lm, guidance);
var next := helpers.UnconstrainedStep(lm, prompt, generated);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.OpenConstrainedSpan(lm, generated);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.EnterObservedConstrainedSpan(lm, generated);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrained, next);
var generated, insideConstrainedOut, currentConstrainedOut := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrained);
var generated, insideConstrainedOut, currentConstrainedOut, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrained);
var next := helpers.ConstrainedStep(lm, parser, prompt, currentConstrained, eosToken);
var next, success := helpers.DeadEndAvoidingStep(lm, parser, prompt, generated, eosToken, maxRetries);
var next, wasConstrained := helpers.ConfidenceGatedStep(lm, parser, prompt, currentConstrained, eosToken);
helpers.SafeBoostTokenLogits(lm, tokens, amount);
helpers.SafePenalizeTokenLogits(lm, tokens, amount);
var next := helpers.SafeBoostedConstrainedStep(lm, parser, prompt, currentConstrained, tokensToBoost, 4.0, eosToken);
var next := helpers.SafePenalizedConstrainedStep(lm, parser, prompt, currentConstrained, tokensToPenalize, 4.0, eosToken);
var next := helpers.SafeRepetitionPenaltyStep(lm, parser, prompt, currentConstrained, generated, 2.0, eosToken);
var next := helpers.SafeTemperatureConstrainedStep(lm, parser, prompt, currentConstrained, 0.8, eosToken);
var next, usedFallback := helpers.SafeSoftConstrainedStep(lm, parser, prompt, currentConstrained, 8.0, eosToken);
var next := helpers.GroupBoostedConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, 4.0, eosToken);
helpers.BoostValidGroups(lm, parser, currentConstrained, groups, amount);
var next := helpers.AdaptiveConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, 4.0, 12, eosToken);
var nextPen := helpers.AdaptiveConstrainedStepWithPenalties(lm, parser, prompt, currentConstrained, validTokenGroups, 4.0, penaltyTokens, 4.0, 12, eosToken);
var gap := helpers.GetLogitGap(lm);
var topK := helpers.GetTopKTokens(lm, k);
helpers.MaskTokensInPrefix(lm, generated);
var snap := helpers.SaveLogitsSnapshot(lm);
helpers.RestoreLogitsSnapshot(lm, snap);
var candTok, candPre, hitComplete, hitEos, stepsUsed := helpers.SpeculativeConstrainedRollout(lm, parser, prompt, currentConstrained, numSpecSteps, eosToken);
var generatedOut, stoppedOnOpenSpan, stoppedOnEos, stepsUsed := helpers.UnconstrainedChunk(lm, prompt, generated, maxChunkTokens, openSpanToken, eosToken);
var currentOut, hitEos, stepsUsed := helpers.ConstrainedSymbol(lm, parser, constrainedPrompt, currentConstrained, stepTokenBudget, eosToken);
var generatedOut, currentOut, hitEos, stepsUsed := helpers.ConstrainedSymbolInGenerated(lm, parser, constrainedPrompt, generated, currentConstrained, stepTokenBudget, eosToken);
var nextSoft, softOk := helpers.SoftConstrainedStep(lm, parser, prompt, currentConstrained, boostAmount, eosToken);
var nextPenRaw := helpers.PenalizedConstrainedStep(lm, parser, prompt, currentConstrained, tokensToPenalize, penaltyAmount, eosToken);
var nextBoostRaw := helpers.BoostedConstrainedStep(lm, parser, prompt, currentConstrained, tokensToBoost, boostAmount, eosToken);
var nextRep := helpers.RepetitionPenaltyStep(lm, parser, prompt, currentConstrained, generated, penaltyAmount, eosToken);
var nextTemp := helpers.TemperatureConstrainedStep(lm, parser, prompt, currentConstrained, temperature, eosToken);
var rolloutGen, rolloutSteps, rolloutEos := helpers.RolloutConstrainedWithPenalties(lm, parser, prompt, startPrefix, budget, penalties, penaltyAmount, eosToken);
var freeGenerated := helpers.UnconstrainedGeneration(lm, prompt, maxSteps);
var constrainedGenerated, terminatedByEos := helpers.ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken);
var craneGenerated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, minReasoningSteps, eosToken);
var topTok := helpers.GetHighestLogitToken(lm);
var logitOne := helpers.GetTokenLogit(lm, token);
helpers.ScaleAllLogits(lm, scalar);
helpers.BoostTokenLogits(lm, tokensInVocab, amount);
helpers.PenalizeTokenLogits(lm, tokensInVocab, amount);
var subCount := CSDHelpers.CountSubstring(text, sub);
var s := CSDHelpers.PrefixToString(prefix);
var between := CSDHelpers.ExtractContentBetweenDelimiters(text, startDelim, endDelim);
var seenInPrompt := helpers.PrefixAppearsInPrompt(lm, prefix);
var promptResemblance: real := helpers.PrefixResemblesPromptExamples(lm, prefix);
var anyInGroup := helpers.GroupHasValidMember(parser, prefix, group);
var rolledGen, rolledCurrent := helpers.RollbackConstrainedSpan(parser, stablePrefix, generated, currentConstrained);
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
var rolled := CSDHelpers.RollbackToValidPrefix(parser, constrainedPrefix);
var rolled := CSDHelpers.RollbackToCompletePrefix(parser, constrainedPrefix);
var generatedOut, currentOut := helpers.RollbackConstrainedSpan(parser, stablePrefix, generated, currentConstrained);
var generatedOut, currentOut := helpers.RollbackConstrainedSuffix(parser, generated, currentConstrained);
var generatedOut, currentOut := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrained);
var generatedOut, currentOut := helpers.RollbackAndContinue(lm, parser, prompt, generated, currentConstrained, eosToken, maxSteps, closeReserve, maxRetries);
var flat := CSDHelpers.FlattenTokenGroups(validTokenGroups);
var groupIdx := CSDHelpers.GroupContaining(validTokenGroups, token);
var prevTok, foundPrev := helpers.LastTokenBefore(generated, ">>");
var occ := CSDHelpers.CountTokenOccurrences(generated, tok);
var occPrefix := CSDHelpers.OccurrencesInRange(generated, tok, hi);
var since := CSDHelpers.TokensSinceLastOccurrence(generated, tok);
var following := CSDHelpers.ExtractAfterKeyword(prefix, keyword);
var intersection := CSDHelpers.IntersectTokenSets(a, b);
var difference := CSDHelpers.SubtractTokenSets(a, b);
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

### Task prompt guidance

- `helpers.AppendTaskGuidance(lm, guidance)`
  Role: append a CSD-chosen guidance block to the evaluator's existing task
  prompt before generation begins.
  Mechanics: forwards `guidance` to the runtime LM wrapper. The evaluator keeps
  its normal prompt, examples, schema, question, and output contract; the
  guidance is appended as an extra block. Runtime semantics are append-only and
  first-call-wins for the current CSD invocation; empty guidance is ignored.
  Cost: +0.
  Control profile: prompt policy only. Call it once at method start, after
  output initialization and before the first LM generation helper.

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

- `helpers.UnconstrainedGeneration(lm, prompt, maxSteps)`
  Role: bounded free-LM generation helper.
  Mechanics: returns exactly `maxSteps` tokens sampled without parser control.
  Cost: +`maxSteps`.
  Control profile: full free generation with no EOS special case.

- `helpers.CraneGeneration(lm, parser, prompt, maxSteps, minReasoningSteps, eosToken)`
  Role: CRANE-style baseline generation with free text outside constrained spans
  and parser-aware decoding inside observed spans.
  Mechanics: emits free tokens until constrained-span state is reached, then
  uses confidence-gated parser control inside that span.
  Cost: at most +`maxSteps`.
  Control profile: free outer continuation plus parser fallback inside spans.

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
  Role: one inside-span token choice for non-exact spans that uses hard parser
  control only when the LM's current top token would not preserve parser
  validity.
  Mechanics: calls the LM, reads the highest-logit token, returns it directly
  if it is EOS or keeps `currentConstrained + [token]` parser-valid; otherwise
  hard-masks to parser-valid next tokens plus EOS and samples from that mask.
  Returns `wasConstrained == true` only on the hard-mask fallback path.
  Cost: +1 token-step, including EOS.
  Control profile: LM-preferred continuation when already parser-compatible,
  hard parser fallback otherwise; avoid using this helper for exact visible
  spans that must remain fully hard-controlled.

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

- `helpers.ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken)`
  Role: bounded parser-valid generation from an empty constrained prefix.
  Mechanics: loops `ConstrainedStep` until parser completeness, EOS, or budget.
  Cost: +`|generated|`, plus one additional step when terminated by EOS.
  Control profile: full hard-parser generation.

- `helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrained)`
  Role: exit a complete constrained span.
  Mechanics: requires `parser.IsCompletePrefix(currentConstrained)`, appends
  visible `">>"` unless already emitted by the constrained grammar, exits
  constrained mode, and clears `currentOut`.
  Cost: +1 token-step for the close action.
  Control profile: direct delimiter/state control gated by parser completeness.

- `helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrained)`
  Role: close the constrained span only if it already holds a complete parse.
  Mechanics: checks `parser.IsCompletePrefix(currentConstrained)` internally; when
  complete it delegates to `CloseConstrainedSpan` (appends `">>"` unless already
  present, exits constrained mode, clears `currentOut`) and returns `closed == true`;
  when not yet complete it leaves state unchanged and returns `closed == false`.
  Cost: +1 token-step when it closes; +0 when it leaves the span open.
  Control profile: completeness-gated close that needs no caller-side proof of completeness; safe to call speculatively each step (no-op until the span parses) and branch on `closed`.

- `helpers.CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrained, eosToken, budget)`
  Role: bring an open constrained span to a completable state and close it within a step budget.
  Mechanics: generates forward inside the span (dead-end-aware), tracking the longest prefix that
  parses as complete; reserves one step and emits `">>"` at that longest complete point. If no
  completable state is reached within `budget`, the span is left open.
  Requires: `lm.ValidTokensIdsLogits()`, `parser.IsValidPrefix(currentConstrained)`,
  `|currentConstrained| <= |generated|`, `eosToken in lm.Tokens`, `">>" in lm.Tokens`.
  Returns: `(generatedOut, insideOut, currentOut)` with `!insideOut ==> currentOut == []`,
  `insideOut ==> parser.IsValidPrefix(currentOut)`, `|generatedOut| <= |generated| + budget`,
  `cost <= old(cost) + budget`, and `cost >= old(cost)`.
  Cost: ≤ `budget` token-steps.
  Control profile: budget-bounded completeness-tracking close that needs no caller-side proof of completeness.

- `helpers.ManagedStep(lm, parser, prompt, generated, insideConstrained, currentConstrained, validTokenGroups, boostAmount, narrowThreshold, eosToken)`
  Role: one self-contained free-or-constrained decode step with delimiter/state
  management built in.
  Mechanics: outside a span, takes one `UnconstrainedStep` and enters constrained
  mode when the full rendered output ends in `"<<"`; inside a span, closes when `currentConstrained` is
  complete, otherwise takes one `AdaptiveConstrainedStep` and appends it.
  Returns `(generatedOut, insideOut, currentOut, done)` where `done` means EOS or
  span close was reached.
  Requires: same parser-valid/current-state shape as the lower-level span helpers,
  `"<<", ">>", eosToken in lm.Tokens`, and bounded non-negative `boostAmount`.
  Cost: exactly +1 token-step on every path.
  Control profile: proof-friendly single-step state machine for strategies that
  need to manage spans without hand-writing delimiter and parser-state branches.

- `helpers.GenerateWithManagedSpan(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, validTokenGroups, boostAmount, narrowThreshold, eosToken)`
  Role: bounded full decode loop with free preamble, constrained span handling,
  completeness-gated close, and proof obligations discharged inside the helper.
  Mechanics: loops up to `maxSteps`; outside a span it samples freely until EOS
  or the full rendered output ends in `"<<"`; inside a span it closes when complete, otherwise advances with
  `AdaptiveConstrainedStep`. The caller receives `(generated,
  insideConstrainedOut, currentConstrainedOut)` and does not need to prove a
  custom loop.
  Requires: same parser-valid/current-state shape as `ManagedStep`, `"<<", ">>",
  eosToken in lm.Tokens`, and bounded non-negative `boostAmount`.
  Cost: at most +`maxSteps`.
  Control profile: high-level span manager for cold strategies that are losing
  budget or correctness in hand-written free/inside-span loops.

- `helpers.GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, prefixBudget, validTokenGroups, boostAmount, narrowThreshold, eosToken)`
  Role: bounded full decode loop that hard-caps the unconstrained preamble before
  forcing entry into a constrained span.
  Mechanics: like `GenerateWithManagedSpan`, but while outside a span it allows
  at most `prefixBudget` free steps. When `prefixBudget < maxSteps`, the next
  loop iteration calls `OpenConstrainedSpan` so the remaining budget can be
  spent on parser-controlled content and closing. When `prefixBudget == maxSteps`,
  the free-token budget consumes the whole loop and
  no forced opening occurs.
  Requires: same requirements as `GenerateWithManagedSpan`, plus
  `prefixBudget <= maxSteps`.
  Cost: at most +`maxSteps`.
  Control profile: output-budget guard for failures where the model stays in
  free text too long or enters constrained mode too late.

- `helpers.DeadEndAvoidingStep(lm, parser, prompt, generated, eosToken, maxRetries)`
  Role: one constrained token step that refuses an immediately invalid or dead
  parser prefix and resamples from the same logits up to `maxRetries` times.
  Mechanics: generates logits once, masks grammar-invalid choices, and masks each
  sampled dead-end token before trying again. Returns `(next, success)` so the
  caller can roll back when no usable continuation is found.
  Cost: exactly +1 token-step.
  Control profile: bounded one-token lookahead for grammars whose runtime mask
  can admit a token with no valid continuation.

### Soft preferences and group-aware constrained decoding

- `helpers.GroupBoostedConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, amount, eosToken)`
  Role: one parser-valid token choice with caller-supplied group preferences.
  Mechanics: calls the LM, boosts groups containing parser-valid members, then
  hard-masks to parser-valid next tokens plus EOS.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with soft preference among legal choices.

- `helpers.BoostValidGroups(lm, parser, prefix, groups, amount)`
  Role: apply the same per-group soft boost used inside group-boosted steps, without sampling.
  Mechanics: for each `groups[i]` such that `GroupHasValidMember` is true at `prefix`, calls `SafeBoostTokenLogits(lm, groups[i], amount)` (non-vocabulary tokens in the group are ignored).
  Cost: +0 (no `GenerateLogits` / `ChooseNextToken` in this helper).
  Control profile: logit shaping only; call after `GenerateLogits` if logits must reflect the current prefix.

- `helpers.AdaptiveConstrainedStep(lm, parser, prompt, currentConstrained, validTokenGroups, amount, narrowThreshold, eosToken)`
  Role: one parser-valid token choice with group preferences applied only at
  narrower parser states.
  Mechanics: same hard mask as `ConstrainedStep`; group boosts are applied only
  when `parser.ValidNextTokenCount(currentConstrained) <= narrowThreshold`.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with conditional soft preference.

- `helpers.AdaptiveConstrainedStepWithPenalties(lm, parser, prompt, currentConstrained, boostGroups, boostAmount, penaltyTokens, penaltyAmount, narrowThreshold, eosToken)`
  Role: same adaptive group boosts as `AdaptiveConstrainedStep`, plus safe token
  penalties before the hard mask.
  Mechanics: `GenerateLogits`, conditional `BoostValidGroups`, `SafePenalizeTokenLogits`,
  `MaskValidNextAndEos`, `ChooseNextToken`.
  Cost: +1 token-step, including EOS.
  Control profile: hard parser control with conditional boosts and penalties.

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

- `helpers.SafeBoostTokenLogits(lm, tokens, amount)`
  Role: raise soft preference for a caller-supplied token set in the current logits.
  Mechanics: filters `tokens` through `lm.Tokens`, then adds to their existing
  logits. It does not call the LM, sample, append output, or inspect the parser.
  Cost: +0.
  Control profile: soft logit preference only; relevant only to later choices
  that read the modified logits rather than regenerating fresh logits first.

- `helpers.SafePenalizeTokenLogits(lm, tokens, amount)`
  Role: lower soft preference for a caller-supplied token set in the current logits.
  Mechanics: filters `tokens` through `lm.Tokens`, then subtracts from their
  existing logits. It does not call the LM, sample, append output, or inspect
  the parser.
  Cost: +0.
  Control profile: soft logit preference only; relevant only to later choices
  that read the modified logits rather than regenerating fresh logits first.

- `helpers.GetLogitGap(lm)`
  Role: measure spread between the top two **unmasked** vocabulary logits.
  Mechanics: single scan over `lm.Logits` with `lm.IsMasked`; returns `0.0` if
  fewer than two unmasked positions exist.
  Cost: +0.

- `helpers.GetTopKTokens(lm, k)`
  Role: return the `k` vocabulary tokens with highest current logits (no LM call).
  Mechanics: greedy index selection with lower-index tie-break; requires
  `1 <= k <= |lm.Tokens|`.
  Cost: +0.

- `helpers.MaskTokensInPrefix(lm, prefix)`
  Role: hard-mask every vocabulary token that appears anywhere in `prefix`.
  Mechanics: walks `prefix` and calls `lm.MaskToken` for in-vocabulary entries.
  Cost: +0.

- `helpers.SpeculativeConstrainedRollout(lm, parser, prompt, constrainedPrefix, numTokens, eosToken)`
  Role: run up to `numTokens` `ConstrainedStep` calls from `constrainedPrefix`, then restore the entry logits exactly
  so the caller can inspect the candidate without
  committing logits state (cost still includes the speculative forward steps).
  Mechanics: internal `SaveLogitsSnapshot` / `RestoreLogitsSnapshot`;
  `candidatePrefix == constrainedPrefix + candidateTokens`; `hitComplete` exactly reports
  whether `candidatePrefix` is complete; `hitEos` means EOS consumed one step without
  being appended to `candidateTokens` and cannot coincide with `hitComplete`.
  Cost: +`stepsUsed` token-steps (`stepsUsed <= numTokens`).

### Parser queries, repair, and context extraction

- `helpers.ValidTokenCount(parser, currentConstrained)`
  Role: inspect parser branching at the current constrained prefix.
  Mechanics: returns the valid-next-token count; no LM call and no state change.
  Cost: +0.
  Control profile: parser information only.

- `helpers.DeadEndDetection(parser, currentConstrained, minValidCount)`
  Role: detect whether parser branching is below a caller-supplied threshold.
  Mechanics: returns a thresholded boolean from the valid-next-token count; no
  LM call and no state change.
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

- `CSDHelpers.RollbackToValidPrefix(parser, prefix)` (static)
  Role: shorten `prefix` from the end until it is a valid, non-dead-end parse
  prefix (or empty).
  Mechanics: drops trailing tokens while the prefix is not `IsValidPrefix` or is
  a dead end (`IsDeadPrefix`); returns the longest prefix the parser can still
  extend, or empty if none exists.
  Cost: +0.
  Control profile: parser repair by deletion.

- `CSDHelpers.RollbackToCompletePrefix(parser, prefix)` (static)
  Role: shorten `prefix` from the end until it is a complete parse or empty.
  Mechanics: drops trailing tokens while the prefix is not `IsCompletePrefix`;
  returns the longest complete prefix, or empty if none exists.
  Cost: +0.
  Control profile: parser repair by deletion.

- `helpers.RollbackConstrainedToComplete(parser, generated, currentConstrained)`
  Role: repair active constrained state by shortening the constrained suffix to
  a complete parse.
  Mechanics: computes the stable prefix from the current suffix length, rolls
  back only `currentConstrained` until `IsCompletePrefix` (or empty), and
  reconstructs `generatedOut`.
  Cost: +0.
  Control profile: parser repair by deletion.

- `helpers.RollbackAndContinue(lm, parser, prompt, generated, currentConstrained, eosToken, maxSteps, closeReserve, maxRetries)`
  Role: repair active constrained state by rolling the constrained suffix back to
  a complete parse and then regenerating forward from that point.
  Mechanics: computes the stable prefix from the current suffix length, rolls
  `currentConstrained` back to its last `IsCompletePrefix` (or empty), then steps
  the LM forward, avoiding dead-end tokens (up to `maxRetries` retries per step)
  and tracking the longest complete prefix reached. Stops after at most
  `maxSteps - closeReserve` steps. `currentOut` is `IsCompletePrefix` or empty;
  `generatedOut` is the stable prefix followed by `currentOut`.
  The constrained span is left open — no closing delimiter is emitted — and the
  `closeReserve` steps withheld from the step budget remain available for closing
  it afterward.
  Requires `closeReserve <= maxSteps` and `|currentConstrained| <= |generated|`.
  Cost: at most +`(maxSteps - closeReserve)`.
  Control profile: parser repair by deletion plus dead-end-avoiding regeneration.

- `helpers.RegenerateUnitOnCheckFailure(lm, parser, prompt, currentConstrained, eosToken, maxStepsPerUnit, maxRetries, maxRollbackBudget, allowedUnits)`
  Role: generate a constrained span unit-by-unit, rewinding and resampling any
  unit whose rendered text is not in a caller-supplied allowed set.
  When to use: when evaluation shows outputs that are syntax-valid but score
  incorrect — i.e. the failure is in WHICH units the model chose, not in
  structure. If outputs are already well-formed (grammar-valid) yet wrong, the
  problem is unit selection; this helper directly addresses that by checking each
  completed grammar unit against a set of acceptable values and resampling on
  mismatch.
  How to use: build `allowedUnits` from the identifiers or names that appear
  in the per-example prompt context (e.g. scan `prompt` for recognizable tokens
  using `CSDHelpers.PrefixToString` or by extracting from the rendered input
  string). Pass this as a `seq<string>`. Start with conservative retry values
  such as `maxStepsPerUnit := 20`, `maxRetries := 3`, `maxRollbackBudget := 10`.
  Example call shape:
    var allowed: seq<string> := /* names extracted from prompt context */;
    var result := helpers.RegenerateUnitOnCheckFailure(
        lm, parser, prompt, currentConstrained, eosToken,
        20, 3, 10, allowed);
  If `allowedUnits` is empty (e.g. because no names were found in context), the
  check is disabled and the helper degrades gracefully to plain dead-end-avoiding
  generation — so it is always safe to call.
  Mechanics: generates tokens via `DeadEndAvoidingStep`. At each grammar-unit
  boundary (`parser.IsCompletePrefix` becomes true), it renders the new unit text
  and checks it against `allowedUnits`. On a failed check and with remaining retry
  budget, it rolls back to the last accepted checkpoint, penalizes the rejected
  first token, and regenerates from there. After `maxRetries` failures on one
  unit, the unit is accepted to preserve termination. Returns `resultConstrained:
  Prefix`, which is always parser-valid.
  Cost: bounded by `(maxRetries + 1) * maxStepsPerUnit` total steps; degrades to
  plain generation when `allowedUnits` is empty or the budget is exhausted.
  Control profile: unit-level rollback-and-resample with bounded retry and
  budget; grammar-driven boundary detection; caller-supplied allowed-unit set.

- `helpers.RegenerateUnitOnGroundingFailure(lm, parser, prompt, currentConstrained, eosToken, budget, maxRetries, maxRollbackBudget)`
  Role: generate a constrained span unit-by-unit, rewinding and resampling any
  completed unit whose identifier-like tokens are not grounded in the prompt
  context.
  Mechanics: generates tokens via dead-end-avoiding steps. A unit boundary is
  detected when `parser.CompletedSchemaSymbolCount` increases, meaning a table
  or column symbol completed. `lm.FirstUngroundedIdentifierTokenIdx` locates the
  actual unsupported identifier inside that unit. With retry budget remaining,
  the helper rolls back to the last accepted checkpoint and persistently
  penalizes the rejected identifier at its own token position before regenerating.
  After `maxRetries` failures on one unit, it accepts the unit to preserve
  termination. Returns `resultConstrained: Prefix`, always parser-valid.
  Cost: at most +`budget` token-steps.
  Control profile: unit-level rollback-and-resample with bounded retry and
  one flat total budget; schema-symbol boundary detection; prompt-grounded
  identifier acceptance test.

- `lm.SpanGrounded(text)` (predicate)
  Returns true iff every identifier-like token in `text` appears in the support
  set extracted from the per-example prompt context; returns true when the prompt
  contains no recognizable support set (so it is always safe to call). Identifier-
  like tokens exclude quoted string-literal contents and short alias-like tokens
  (a single letter, or letters followed by digits). Pure with respect to the
  Dafny heap. Fair: the support set is derived only from prompt text, never from
  execution feedback or gold answers.

- `helpers.PrefixAppearsInPrompt(lm, prefix)`
  Role: no-gold duplicate/exemplar check for a generated prefix.
  Mechanics: renders `prefix` and asks whether the normalized span already
  appears in the current prompt context, including prompt examples and rolling
  suffix text visible to the model. It uses exact normalized span matching, not
  substring search.
  Cost: +0.
  Fair: reads only prompt-visible text; never reads gold labels, scorer state,
  evaluator results, or class-specific win rules.

- `helpers.PrefixResemblesPromptExamples(lm, prefix)`
  Role: no-gold resemblance score for a generated prefix.
  Mechanics: renders `prefix` and returns a real number in [0,1] measuring how
  structurally similar it is to the example spans shown in the current prompt
  (the same prompt-visible examples `PrefixAppearsInPrompt` inspects). Similarity
  is computed with generic tooling. Returns 0.0 when the prompt shows no examples
  or the prefix cannot be parsed. The strategy chooses its own threshold.
  Cost: +0.
  Fair: reads only prompt-visible examples; never reads gold labels, scorer
  state, held-out data, evaluator results, or class-specific strategy advice.

- `helpers.LastTokenBefore(generated, sep)` and
  `CSDHelpers.ExtractAfterKeyword(prefix, keyword)`
  Role: read lightweight context from existing generated tokens.
  Mechanics: scan token sequences and return matching context tokens; no LM
  call and no state change.
  Cost: +0.
  Control profile: context information only.

- `CSDHelpers.CountTokenOccurrences(prefix, target)` (static)
  Role: count how many times `target` appears as a token element of `prefix`.
  Cost: +0.

- `CSDHelpers.OccurrencesInRange(prefix, target, hi)` (static function)
  Role: count how many times `target` appears before index `hi`.
  Contract: requires `hi <= |prefix|`.
  Cost: +0.

- `CSDHelpers.TokensSinceLastOccurrence(prefix, target)` (static)
  Role: distance in tokens from the end of `prefix` back to the last occurrence
  of `target`; returns `|prefix|` if `target` never occurs.
  Cost: +0.

- `CSDHelpers.FlattenTokenGroups`, `CSDHelpers.GroupContaining`,
  `CSDHelpers.IntersectTokenSets`, and `CSDHelpers.SubtractTokenSets`
  Role: transform token sets or groups.
  Mechanics: operate on sequences only; no LM call, parser query, output append,
  or state transition.
  Cost: +0.
  Control profile: token-set bookkeeping only.

- `helpers.GroupHasValidMember(parser, prefix, group)`
  Role: test whether any token in `group` is parser-valid at `prefix`.
  Mechanics: linear scan over `group`; no LM call or logit change.
  Cost: +0.

- `helpers.RollbackConstrainedSpan(parser, stablePrefix, generated, currentConstrained)`
  Role: repair constrained text when `generated == stablePrefix + currentConstrained`.
  Mechanics: rolls back `currentConstrained` with `RollbackToValidPrefix`, then
  reattaches `stablePrefix`.
  Cost: +0.

- `helpers.SoftConstrainedStep(lm, parser, prompt, constrainedPrefix, boostAmount, eosToken)`
  Role: one step that boosts grammar-valid logits (and EOS) then samples without a hard mask.
  Mechanics: `BoostValidNextAndEos`, `ChooseNextTokenUnconstrained`; `isValid` reports
  EOS or parser-valid extension.
  Cost: +1 token-step.

- `helpers.PenalizedConstrainedStep`
  Role: `ConstrainedStep` with an explicit penalize list that must already be in `lm.Tokens`.
  Mechanics: same hard mask and postconditions as `ConstrainedStep` after the
  penalty edit.
  Cost: +1 token-step.

- `helpers.BoostedConstrainedStep`
  Role: `ConstrainedStep` with an explicit boost list that must already be in `lm.Tokens`.
  Mechanics: same hard mask and postconditions as `ConstrainedStep` after the
  boost edit.
  Cost: +1 token-step.

- `helpers.BoostTokenLogits`
  Role: direct positive logit nudge for a vocabulary-known token list.
  Mechanics: requires `forall t in list :: t in lm.Tokens`; clamped arithmetic; no LM call.
  Cost: +0.

- `helpers.PenalizeTokenLogits`
  Role: direct logit nudge for a vocabulary-known token list.
  Mechanics: requires `forall t in list :: t in lm.Tokens`; clamped arithmetic; no LM call.
  Cost: +0.

- `helpers.RepetitionPenaltyStep`
  Role: constrained step with repetition penalty on a token bag before masking.
  Mechanics: non-safe variant imposes the same membership proofs as the
  underlying penalize helper.
  Cost: +1 token-step.

- `helpers.TemperatureConstrainedStep`
  Role: constrained step with temperature scaling before masking.
  Mechanics: non-safe variant imposes the same range proofs as the underlying
  scale helper.
  Cost: +1 token-step.

- `helpers.RolloutConstrainedWithPenalties(lm, parser, prompt, startPrefix, totalBudget, penalties, penaltyAmount, eosToken)`
  Role: bounded loop of `SafePenalizedConstrainedStep` until completion, EOS, or budget.
  Mechanics: extends `generatedOut` token by token; `cost` increases by `stepsUsed`.
  Cost: +`stepsUsed` token-steps (each inner step +1).

- `helpers.GetHighestLogitToken(lm)`
  Role: read the argmax from the current logit vector.
  Mechanics: no forward pass; assumes logits already match the intended prefix.
  Cost: +0.

- `helpers.GetTokenLogit(lm, token)`
  Role: read one coordinate from the current logit vector.
  Mechanics: no forward pass; assumes logits already match the intended prefix.
  Cost: +0.

- `helpers.ScaleAllLogits(lm, scalar)`
  Role: multiply every vocabulary logit by a positive scalar with bounds clamping.
  Mechanics: no LM call; use before a sampling helper that reads the same logits.
  Cost: +0.

- `helpers.SaveLogitsSnapshot(lm)`
  Role: copy logits for branching or speculation.
  Mechanics: full-array read; use with a later restore when a speculative path
  should not commit logit state.
  Cost: +0 for snapshot ops alone.

- `helpers.RestoreLogitsSnapshot(lm, snapshot)`
  Role: restore copied logits after branching or speculation.
  Mechanics: full-array write; restores prior logits without changing `cost` by itself.
  Cost: +0 for snapshot ops alone.

- `CSDHelpers.CountSubstring(s, sub)` (static function; same class as instance helpers)
  Role: count non-overlapping occurrences of `sub` in string `s`.
  Cost: +0.

- `CSDHelpers.PrefixToString(prefix)` / `CSDHelpers.ExtractContentBetweenDelimiters(input, startDelim, endDelim)` (static)
  Role: stringify a token prefix or extract delimited substring content per contract.
  Cost: +0.

## LM and `Parser` surface (for calls inside the strategy body)

Strategy code may also invoke `lm` and `parser` members directly when proofs permit.
Summaries align with `synthesis/verify/library/README.md`; the `.dfy` file states full contracts.

- **`lm`:** `GenerateLogits`, `ChooseNextToken`, `ChooseNextTokenUnconstrained`, `GenerateUnconstrainedChunk`,
  `MaskValidNextAndEos`, `BoostValidNextAndEos`, `MaskToken` / `MaskTokens` / `MaskTokensExcept`,
  `IdToToken`, `TokenToId`, logit readers (`IdToLogit`, `TokenToLogit`, `TokensToLogits`,
  `IdsToLogits`), `IsMasked`, `HasUnmaskedToken`.
- **`parser`:** `IsValidPrefix`, `IsCompletePrefix`, `IsDeadPrefix`, `ValidNextTokenCount`, `ValidNextToken`,
  `ValidNextTokens`, `ParseG`.

### Verified rendered-text functions

- `Contains(s, sub)` tests whether plain string `s` contains plain string `sub`.
- `RenderPrefix(prefix)` concatenates token strings into the exact text they render.
- `RenderedEndsWith(prefix, suffix)` tests the rendered text suffix without assuming
  that a visible substring is one tokenizer token. After appending `next` to
  `generated`, prefer `RenderedEndsWith(generated, "<<")` over `next == "<<"`:
  it recognizes a space-prefixed token such as `" <<"` and a delimiter split across tokens.

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


_TOOL_REFERENCE_START = "\n## Available Tools\n"
_PROOF_DISCIPLINE_START = "\n## Proof sketch discipline\n"
_TOOL_REFERENCE_START_INDEX = SYSTEM_PROMPT.index(_TOOL_REFERENCE_START)
_PROOF_DISCIPLINE_START_INDEX = SYSTEM_PROMPT.index(_PROOF_DISCIPLINE_START)
TOOL_REFERENCE = SYSTEM_PROMPT[
    _TOOL_REFERENCE_START_INDEX:_PROOF_DISCIPLINE_START_INDEX
].strip()
SYSTEM_PROMPT = (
    SYSTEM_PROMPT[:_TOOL_REFERENCE_START_INDEX].rstrip()
    + "\n\n"
    + SYSTEM_PROMPT[_PROOF_DISCIPLINE_START_INDEX:].lstrip()
)

_HELPER_CALL_RE = re.compile(
    r"\b(?:helpers|CSDHelpers)\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_HELPER_REF_RE = re.compile(
    r"\b(?:helpers|CSDHelpers)\.([A-Za-z_][A-Za-z0-9_]*)\b"
)
_MENU_PRUNED_HELPERS = {
    "RollbackToValidPrefix",
    "RollbackToCompletePrefix",
    "RollbackConstrainedSpan",
    "RollbackAndRegenerate",
    "RolloutConstrainedWithPenalties",
    "RegenerateUnitOnCheckFailure",
}
_ALL_HELPER_NAMES = (
    set(_HELPER_REF_RE.findall(TOOL_REFERENCE))
    - _MENU_PRUNED_HELPERS
    | {"GenerateWithManagedSpan", "GenerateWithPrefixAndManagedSpan", "ManagedStep"}
)


def _filter_code_fence_lines(text: str, allowed: set[str]) -> str:
    """Drop helper signatures that are outside the active helper contract."""
    lines = []
    for line in text.splitlines():
        helper_names = set(_HELPER_REF_RE.findall(line))
        if helper_names and helper_names.isdisjoint(allowed):
            continue
        lines.append(line)
    return "\n".join(lines).rstrip()


def _iter_helper_doc_blocks(section_body: str):
    block: list[str] = []
    for line in section_body.splitlines():
        if line.startswith("- `"):
            if block:
                yield block
            block = [line]
        elif block:
            block.append(line)
    if block:
        yield block


def _filter_tool_api_reference(text: str, allowed: set[str]) -> str:
    """Keep API reference bullets only for helpers visible this attempt."""
    lm_parser_surface = ""
    lm_parser_header = "\n## LM and `Parser` surface"
    if lm_parser_header in text:
        text, lm_parser_surface = text.split(lm_parser_header, 1)
        lm_parser_surface = lm_parser_header.strip() + lm_parser_surface.rstrip()

    lines = text.splitlines()
    section_starts = [
        index for index, line in enumerate(lines) if line.startswith("### ")
    ]
    if not section_starts:
        parts = [text.rstrip()] if text.rstrip() else []
        if lm_parser_surface:
            parts.append(lm_parser_surface)
        return "\n\n".join(parts)

    intro = "\n".join(lines[:section_starts[0]]).rstrip()
    rendered_sections = []

    for start_index, next_start_index in zip(
        section_starts,
        section_starts[1:] + [len(lines)],
    ):
        header = lines[start_index]
        body = "\n".join(lines[start_index + 1:next_start_index])
        kept_blocks = []
        for block in _iter_helper_doc_blocks(body):
            helper_names = set(_HELPER_REF_RE.findall("\n".join(block)))
            if helper_names and helper_names.issubset(allowed):
                kept_blocks.append("\n".join(block).rstrip())
        if kept_blocks:
            rendered_sections.append(
                header + "\n\n" + "\n\n".join(kept_blocks)
            )

    parts = [intro] if intro else []
    parts.extend(rendered_sections)
    if lm_parser_surface:
        parts.append(lm_parser_surface)
    return "\n\n".join(parts).rstrip()


def _filter_tool_reference(allowed_helpers: list[str]) -> str:
    """Render a reduced tool catalog for the currently allowed helper set."""
    allowed = set(allowed_helpers)
    if "\n## Tool API Reference\n" not in TOOL_REFERENCE:
        return _filter_code_fence_lines(TOOL_REFERENCE, allowed)

    available_tools, api_reference = TOOL_REFERENCE.split(
        "\n## Tool API Reference\n",
        1,
    )
    return (
        _filter_code_fence_lines(available_tools, allowed)
        + "\n\n## Tool API Reference\n\n"
        + _filter_tool_api_reference(api_reference, allowed)
    ).rstrip()


VERIFIED_EXAMPLES = 'The verified examples below are pattern demonstrations, not task-specific recommendations.\nUse them as a palette of mechanisms: span entry, constrained progression,\nclosing/termination, repair, chunking, and preference shaping. Adapt or combine\nonly the parts whose control behavior matches the current task contract and\nmeasured failures.\n\nWhen the output contract uses visible delimiter spans, keep each span as short\nas possible, open it only when the exact content is known, and close it\nimmediately after the final token of that exact content.\nKeep all intermediate reasoning outside visible spans; a visible span should\ncontain only the final exact expression or answer content, not setup or prose.\nIf the task allows it, use only one visible span for the final answer rather\nthan emitting multiple visible spans throughout the solution.\nFor visible spans that contain exact content, prefer parser-valid token steps\nover free-form chunks, and avoid repetition penalties that can amplify valid\nsymbol loops.\nFor exact visible spans, use hard parser-controlled helpers such as\n`ConstrainedStep`; do not use `ConfidenceGatedStep` inside exact spans.\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Guided-adaptive CSD. The strategy uses an append-only prompt-guidance\n// block to steer the model, then adapts the decoder: it uses group boosting\n// while the active prefix is narrow, and switches to penalty-aware constrained\n// decoding once the prefix has grown past the initial region. The keyword\n// groups, penalty tokens, and narrow threshold are caller-provided parameters.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: AppendTaskGuidance leaves generated state unchanged.\n//   Outside the span, the implication is vacuous unless "<<" is observed, which\n//   resets currentConstrainedOut to the valid empty prefix. CloseConstrainedSpan\n//   exits constrained mode. GroupBoostedConstrainedStep and\n//   AdaptiveConstrainedStepWithPenalties return EOS or parser-valid tokens, and\n//   AppendConstrainedToken preserves parser validity.\n// progress: AppendTaskGuidance costs 0. Every later branch consumes one step\n//   and appends at most one visible token, so the output-length bound remains\n//   linear in steps. The adaptive branch only changes which parser-valid helper\n//   is used.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar guidance: string := "Generate exactly one task-appropriate output. No explanation or Markdown. Follow the task\'s declared output contract exactly.";\nhelpers.AppendTaskGuidance(lm, guidance);\n\n// Caller-provided parameters (in practice these would be passed via the task description or synthesis context)\nvar keywordGroups: seq<seq<Token>> := validTokenGroups; // Default to caller-supplied groups\nvar penaltyTokens: seq<Token> := []; // Caller-provided tokens to penalize (e.g., early terminators)\nvar narrowThreshold: nat := 10;\nvar steps: nat := 0;\nvar phase: nat := 0; // 0 = initial narrow phase, 1 = post-penalty phase\n\nwhile steps < maxSteps\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  invariant phase == 0 || phase == 1\n  decreases maxSteps - steps\n{{\n  if !insideConstrainedOut {{\n    var next := helpers.UnconstrainedStep(lm, prompt, generated);\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      generated := generated + [next];\n      if RenderedEndsWith(generated, "<<") {{\n        insideConstrainedOut := true;\n        currentConstrainedOut := [];\n        phase := 0;\n      }}\n    }}\n  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{\n    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(\n      lm, parser, generated, currentConstrainedOut\n    );\n    generated := closedGenerated;\n    insideConstrainedOut := closedInside;\n    currentConstrainedOut := closedCurrent;\n    steps := steps + 1;\n  }} else {{\n    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];\n    var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);\n    var next := eosToken;\n    if phase == 0 && validCount <= narrowThreshold {{\n      var groups := keywordGroups + validTokenGroups;\n      next := helpers.GroupBoostedConstrainedStep(\n        lm, parser, constrainedPrompt, currentConstrainedOut, groups, 6.0, eosToken\n      );\n      if validCount > narrowThreshold {{\n        phase := 1;\n      }}\n    }} else if phase == 1 && |penaltyTokens| > 0 {{\n      next := helpers.AdaptiveConstrainedStepWithPenalties(\n        lm, parser, constrainedPrompt, currentConstrainedOut,\n        validTokenGroups, 4.0, penaltyTokens, 5.0, 8, eosToken\n      );\n    }} else {{\n      next := helpers.AdaptiveConstrainedStep(\n        lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, narrowThreshold, eosToken\n      );\n    }}\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(\n        lm, parser, generated, currentConstrainedOut, next\n      );\n      generated := appendedGenerated;\n      insideConstrainedOut := appendedInside;\n      currentConstrainedOut := appendedCurrent;\n    }}\n  }}\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Simple delimiter-triggered CSD. Generate freely until "<<" appears, then\n// constrain; each constrained step calls CloseSpanIfComplete, which emits ">>"\n// once the parser accepts the span and is a no-op otherwise.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: In the unconstrained branch we only flip insideConstrainedOut\n//   to true when next renders with suffix "<<", and we set currentConstrainedOut := [] which is\n//   a valid prefix. In the constrained branch, CloseSpanIfComplete either closes\n//   the span (sets insideConstrainedOut to false, making the implication vacuous,\n//   and clears currentConstrainedOut to []) or leaves generated/inside/current\n//   unchanged (no-op). In the no-op path, ConstrainedStep plus AppendConstrainedToken\n//   preserves parser validity when IsTokenValidNext holds.\n// progress: Every branch appends at most one token to generated and steps grows\n//   by 1, so |generated| - |generatedPrefix| <= steps <= steps * stepTokenBudget.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar steps: nat := 0;\n\nwhile steps < maxSteps\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  decreases maxSteps - steps\n{{\n  if !insideConstrainedOut {{\n    var next := helpers.UnconstrainedStep(lm, prompt, generated);\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      generated := generated + [next];\n      if RenderedEndsWith(generated, "<<") {{\n        insideConstrainedOut := true;\n        currentConstrainedOut := [];\n      }}\n    }}\n  }} else {{\n    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);\n    steps := steps + 1;\n    if closed {{\n      generated := cg;\n      insideConstrainedOut := ci;\n      currentConstrainedOut := cc;\n    }} else {{\n      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];\n      var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);\n      if next == eosToken {{\n        break;\n      }} else {{\n        var valid := helpers.IsTokenValidNext(parser, currentConstrainedOut, next);\n        if valid {{\n          var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(\n            lm, parser, generated, currentConstrainedOut, next\n          );\n          generated := appendedGenerated;\n          insideConstrainedOut := appendedInside;\n          currentConstrainedOut := appendedCurrent;\n        }}\n      }}\n    }}\n  }}\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Context-triggered CSD. The strategy tracks whether a neutral local marker has\n// recently appeared in the free text. When that marker is seen, the next\n// outside-span action opens a constrained span before returning to ordinary\n// delimiter-triggered behavior. Inside the span, parser validity remains the\n// hard authority.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: Outside the span, the implication is vacuous. The\n//   context-triggered open branch uses OpenConstrainedSpan, which returns\n//   currentConstrainedOut == [], valid by parser.IsValidPrefix([]). If the\n//   free token itself is "<<", we enter with currentConstrainedOut := [].\n//   CloseConstrainedSpan exits constrained mode. ConstrainedStep plus\n//   AppendConstrainedToken preserves validity for non-EOS tokens.\n// progress: UnconstrainedStep, OpenConstrainedSpan, CloseConstrainedSpan, and\n//   ConstrainedStep each consume one step and append at most one token, so\n//   |generated| <= |generatedPrefix| + steps is preserved.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar steps: nat := 0;\nvar markerArmed: bool := false;\nvar markerToken: Token := ":";\n\nwhile steps < maxSteps\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  decreases maxSteps - steps\n{{\n  if !insideConstrainedOut {{\n    if markerArmed {{\n      var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);\n      generated := openedGenerated;\n      insideConstrainedOut := openedInside;\n      currentConstrainedOut := openedCurrent;\n      markerArmed := false;\n      steps := steps + 1;\n    }} else {{\n      var next := helpers.UnconstrainedStep(lm, prompt, generated);\n      steps := steps + 1;\n      if next == eosToken {{\n        break;\n      }} else {{\n        generated := generated + [next];\n        if RenderedEndsWith(generated, "<<") {{\n          insideConstrainedOut := true;\n          currentConstrainedOut := [];\n          markerArmed := false;\n        }} else if next == markerToken {{\n          markerArmed := true;\n        }}\n      }}\n    }}\n  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{\n    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(\n      lm, parser, generated, currentConstrainedOut\n    );\n    generated := closedGenerated;\n    insideConstrainedOut := closedInside;\n    currentConstrainedOut := closedCurrent;\n    steps := steps + 1;\n  }} else {{\n    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];\n    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(\n        lm, parser, generated, currentConstrainedOut, next\n      );\n      generated := appendedGenerated;\n      insideConstrainedOut := appendedInside;\n      currentConstrainedOut := appendedCurrent;\n    }}\n  }}\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Group-aware constrained CSD. Generate freely until "<<" appears, then use\n// caller-supplied token groups as a soft preference while the parser remains\n// the hard validity authority.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: Outside the span, the implication is vacuous unless next is\n//   "<<", in which case currentConstrainedOut := [] is valid by precondition.\n//   CloseConstrainedSpan flips insideConstrainedOut to false. In the active\n//   constrained branch, GroupBoostedConstrainedStep returns either EOS or a\n//   parser-valid next token, and AppendConstrainedToken preserves validity.\n// progress: Every branch appends at most one token and steps grows by 1, so\n//   |generated| <= |generatedPrefix| + steps is preserved.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar steps: nat := 0;\n\nwhile steps < maxSteps\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  decreases maxSteps - steps\n{{\n  if !insideConstrainedOut {{\n    var next := helpers.UnconstrainedStep(lm, prompt, generated);\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      generated := generated + [next];\n      if RenderedEndsWith(generated, "<<") {{\n        insideConstrainedOut := true;\n        currentConstrainedOut := [];\n      }}\n    }}\n  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{\n    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(\n      lm, parser, generated, currentConstrainedOut\n    );\n    generated := closedGenerated;\n    insideConstrainedOut := closedInside;\n    currentConstrainedOut := closedCurrent;\n    steps := steps + 1;\n  }} else {{\n    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];\n    var next := helpers.GroupBoostedConstrainedStep(\n      lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, eosToken\n    );\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(\n        lm, parser, generated, currentConstrainedOut, next\n      );\n      generated := appendedGenerated;\n      insideConstrainedOut := appendedInside;\n      currentConstrainedOut := appendedCurrent;\n    }}\n  }}\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Top-candidate constrained CSD. Inside a span, query a small ranked set of\n// parser-valid candidates and append the first non-EOS candidate if available.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: Enter only when the unconstrained token renders with suffix "<<",\n//   at which point currentConstrainedOut := [] (valid). Complete-prefix branch\n//   flips insideConstrainedOut to false via CloseConstrainedSpan. TopValidCandidates\n//   returns only EOS or valid-next tokens; after excluding EOS, AppendConstrainedToken\n//   preserves validity.\n// progress: Every branch appends at most one token; attempts grows by 1, so\n//   |generated| <= |generatedPrefix| + attempts * stepTokenBudget holds.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar attempts: nat := 0;\n\nwhile attempts < maxSteps\n  invariant 0 <= attempts <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + attempts\n  decreases maxSteps - attempts\n{{\n  if !insideConstrainedOut {{\n    var next := helpers.UnconstrainedStep(lm, prompt, generated);\n    attempts := attempts + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      generated := generated + [next];\n      if RenderedEndsWith(generated, "<<") {{\n        insideConstrainedOut := true;\n        currentConstrainedOut := [];\n      }}\n    }}\n  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{\n    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(\n      lm, parser, generated, currentConstrainedOut\n    );\n    generated := closedGenerated;\n    insideConstrainedOut := closedInside;\n    currentConstrainedOut := closedCurrent;\n    attempts := attempts + 1;\n  }} else {{\n    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];\n    var candidates := helpers.TopValidCandidates(\n      lm, parser, constrainedPrompt, currentConstrainedOut, 4, eosToken\n    );\n    var next := candidates[0];\n    if next == eosToken && |candidates| > 1 {{\n      next := candidates[1];\n    }}\n    attempts := attempts + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      assert next in candidates;\n      assert next in parser.ValidNextTokens(currentConstrainedOut);\n      assert parser.IsValidPrefix(currentConstrainedOut + [next]);\n      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(\n        lm, parser, generated, currentConstrainedOut, next\n      );\n      generated := appendedGenerated;\n      insideConstrainedOut := appendedInside;\n      currentConstrainedOut := appendedCurrent;\n    }}\n  }}\n}}\n\ncost := attempts;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Chunked-outside CSD. Outside a constrained span we generate unconstrained\n// tokens in a single multi-token call (`UnconstrainedChunk`) that breaks early\n// on EOS or on the open-span delimiter `"<<"`. Keep the outside chunk budget\n// large enough that the model can emit the task\'s surface prefix before the\n// span opens, but still bounded so it does not run away. Inside a\n// span we decode token by token using the parser the same way the simple\n// delimiter-triggered strategy does.\n// Multi-token chunking amortizes per-token dispatch overhead across the\n// unconstrained region without starving the prefix.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: Outside the span, insideConstrainedOut stays false unless\n//   UnconstrainedChunk reports stoppedOnOpenSpan, in which case we set\n//   currentConstrainedOut := [], a valid prefix by the method precondition\n//   parser.IsValidPrefix([]). In the complete-prefix branch, CloseConstrainedSpan\n//   flips insideConstrainedOut to false, making the implication vacuous. In the\n//   constrained-step branch, AppendConstrainedToken is only invoked after\n//   ConstrainedStep returned a parser-valid next token (or EOS, which breaks).\n// progress: Chunk branch: |generatedOut| <= |generated| + stepsUsed and\n//   steps := steps + stepsUsed, so |new_generated| <= |generatedPrefix| +\n//   steps + stepsUsed = |generatedPrefix| + new_steps. Other branches append\n//   ≤1 token and steps += 1. Linear arithmetic throughout. ✓\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar steps: nat := 0;\n\nwhile steps < maxSteps\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  decreases maxSteps - steps\n{{\n  if !insideConstrainedOut {{\n    var chunkBudget: nat := maxSteps - steps;\n    if chunkBudget > 32 {{\n      chunkBudget := 32;\n    }}\n    var chunkedG, stoppedOpen, stoppedEos, stepsUsed := helpers.UnconstrainedChunk(\n      lm, prompt, generated, chunkBudget, "<<", eosToken\n    );\n    generated := chunkedG;\n    steps := steps + stepsUsed;\n    if stoppedEos {{\n      break;\n    }} else if stoppedOpen {{\n      insideConstrainedOut := true;\n      currentConstrainedOut := [];\n    }}\n  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{\n    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(\n      lm, parser, generated, currentConstrainedOut\n    );\n    generated := closedGenerated;\n    insideConstrainedOut := closedInside;\n    currentConstrainedOut := closedCurrent;\n    steps := steps + 1;\n  }} else {{\n    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];\n    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(\n        lm, parser, generated, currentConstrainedOut, next\n      );\n      generated := appendedGenerated;\n      insideConstrainedOut := appendedInside;\n      currentConstrainedOut := appendedCurrent;\n    }}\n  }}\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Symbol-step CSD. Each outer loop iteration is one "symbol step" — the model\n// is called for up to stepTokenBudget tokens at once, and the longest valid\n// parser prefix of the result is accepted. This aligns generation granularity\n// with the task\'s natural units: SQL keywords, arithmetic expressions, or\n// multi-subword identifiers can be emitted as a unit instead of being forced\n// token by token. Outside a constrained span the strategy generates one token\n// freely. Inside the span it uses ConstrainedSymbol, passing stepTokenBudget\n// as the per-step token allowance. Close as soon as the parser reports the\n// prefix is complete.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: Opening a span sets currentConstrainedOut := [], valid by\n//   precondition. CloseConstrainedSpan flips insideConstrainedOut to false.\n//   ConstrainedSymbol postcondition: parser.IsValidPrefix(symbolOut).\n// progress: steps advances by exactly 1 for UnconstrainedStep and\n//   CloseConstrainedSpan (each appends ≤1 token), and by stepsUsed ≥ 1 for\n//   ConstrainedSymbol (postcondition). So decreases maxSteps - steps always\n//   decreases. The invariant |generated| <= |generatedPrefix| + steps is\n//   linear: each branch adds at most the consumed token budget to visible\n//   output and advances steps by that budget (or 1 for single-token branches). ✓\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar steps: nat := 0;\n\nwhile steps < maxSteps\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  decreases maxSteps - steps\n{{\n  if !insideConstrainedOut {{\n    var next := helpers.UnconstrainedStep(lm, prompt, generated);\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      generated := generated + [next];\n      if RenderedEndsWith(generated, "<<") {{\n        insideConstrainedOut := true;\n        currentConstrainedOut := [];\n      }}\n    }}\n  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{\n    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(\n      lm, parser, generated, currentConstrainedOut\n    );\n    generated := closedGenerated;\n    insideConstrainedOut := closedInside;\n    currentConstrainedOut := closedCurrent;\n    steps := steps + 1;\n  }} else {{\n    var stablePrefix := generated[..|generated| - |currentConstrainedOut|];\n    var constrainedPrompt := prompt + stablePrefix;\n    var symbolBudget: nat := maxSteps - steps;\n    var symbolGenerated, symbolOut, hitEos, stepsUsed := helpers.ConstrainedSymbolInGenerated(\n      lm, parser, constrainedPrompt, generated, currentConstrainedOut, symbolBudget, eosToken\n    );\n    generated := symbolGenerated;\n    currentConstrainedOut := symbolOut;\n    steps := steps + stepsUsed;\n    if hitEos {{\n      break;\n    }}\n  }}\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Adaptive-narrowness CSD. Inside a constrained span, query the parser\'s\n// valid-continuation count and choose between different constrained decoding\n// strategies based on branch factor. Uses caller-provided keyword groups and\n// penalty tokens for the adaptive phases.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: CloseConstrainedSpan makes implication vacuous. In the\n//   tight branch, GroupBoostedConstrainedStep returns a parser-valid next token\n//   and AppendConstrainedToken preserves validity. In the penalty branch,\n//   AdaptiveConstrainedStepWithPenalties returns EOS or parser-valid tokens.\n//   In the default branch, AdaptiveConstrainedStep returns EOS or parser-valid\n//   tokens. All preserve validity via AppendConstrainedToken.\n// progress: Every branch consumes one step and appends at most one visible\n//   token, so |generated| <= |generatedPrefix| + steps is preserved.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar narrowThreshold: nat := 20;\nvar steps: nat := 0;\nvar phase: nat := 0; // 0 = narrow (boost), 1 = penalty phase, 2 = default\nvar keywordGroups: seq<seq<Token>> := validTokenGroups; // Caller-provided\nvar penaltyTokens: seq<Token> := []; // Caller-provided tokens to penalize\n\nwhile steps < maxSteps\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  invariant phase <= 2\n  decreases maxSteps - steps\n{{\n  if !insideConstrainedOut {{\n    var next := helpers.UnconstrainedStep(lm, prompt, generated);\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      generated := generated + [next];\n      if RenderedEndsWith(generated, "<<") {{\n        insideConstrainedOut := true;\n        currentConstrainedOut := [];\n        phase := 0;\n      }}\n    }}\n  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{\n    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(\n      lm, parser, generated, currentConstrainedOut\n    );\n    generated := closedGenerated;\n    insideConstrainedOut := closedInside;\n    currentConstrainedOut := closedCurrent;\n    steps := steps + 1;\n  }} else {{\n    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];\n    var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);\n    var next := eosToken;\n    if phase == 0 && validCount <= narrowThreshold {{\n      var groups := keywordGroups + validTokenGroups;\n      next := helpers.GroupBoostedConstrainedStep(\n        lm, parser, constrainedPrompt, currentConstrainedOut, groups, 6.0, eosToken\n      );\n      if validCount > narrowThreshold {{\n        phase := 1;\n      }}\n    }} else if phase == 1 && |penaltyTokens| > 0 {{\n      next := helpers.AdaptiveConstrainedStepWithPenalties(\n        lm, parser, constrainedPrompt, currentConstrainedOut,\n        validTokenGroups, 4.0, penaltyTokens, 5.0, 8, eosToken\n      );\n      if validCount <= narrowThreshold {{\n        phase := 0;\n      }} else {{\n        phase := 2;\n      }}\n    }} else {{\n      next := helpers.AdaptiveConstrainedStep(\n        lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, narrowThreshold, eosToken\n      );\n    }}\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(\n        lm, parser, generated, currentConstrainedOut, next\n      );\n      generated := appendedGenerated;\n      insideConstrainedOut := appendedInside;\n      currentConstrainedOut := appendedCurrent;\n    }}\n  }}\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Context-tracking CSD. Maintains a strategy-local seq<Token> across loop\n// iterations. After each constrained token append, it queries the span for\n// tokens following a caller-provided keyword via ExtractAfterKeyword. At\n// candidate-selection positions, it intersects parser-valid candidates with\n// that context set and boosts the intersection.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: GroupBoostedConstrainedStep returns either EOS or a\n//   parser-valid next token; the EOS branch breaks. AppendConstrainedToken\n//   preserves validity. CloseConstrainedSpan makes the implication vacuous.\n//   The context variable is never passed to the parser as authority, so it\n//   cannot affect parser_validity.\n// progress: Every branch increments steps by 1 and appends at most one token,\n//   so |generated| <= |generatedPrefix| + steps is preserved throughout.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar semanticContext: seq<Token> := [];\nvar scopeKeyword: Token := ""; // Caller-provided keyword to track (e.g., "FROM", "=", "(")\nvar steps: nat := 0;\n\nwhile steps < maxSteps\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  decreases maxSteps - steps\n{{\n  if !insideConstrainedOut {{\n    var next := helpers.UnconstrainedStep(lm, prompt, generated);\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      generated := generated + [next];\n      if RenderedEndsWith(generated, "<<") {{\n        insideConstrainedOut := true;\n        currentConstrainedOut := [];\n      }}\n    }}\n  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{\n    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(\n      lm, parser, generated, currentConstrainedOut\n    );\n    generated := closedGenerated;\n    insideConstrainedOut := closedInside;\n    currentConstrainedOut := closedCurrent;\n    steps := steps + 1;\n  }} else {{\n    // Update semantic context from accumulated span content\n    if |scopeKeyword| > 0 {{\n      semanticContext := CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, scopeKeyword);\n    }}\n\n    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];\n    var groups := validTokenGroups;\n    if |semanticContext| > 0 {{\n      groups := [semanticContext] + validTokenGroups;\n    }}\n    var next := helpers.GroupBoostedConstrainedStep(\n      lm, parser, constrainedPrompt, currentConstrainedOut, groups, 6.0, eosToken\n    );\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(\n        lm, parser, generated, currentConstrainedOut, next\n      );\n      generated := appendedGenerated;\n      insideConstrainedOut := appendedInside;\n      currentConstrainedOut := appendedCurrent;\n    }}\n  }}\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Rollback-on-stall CSD. Generate freely until "<<" appears. Inside the span,\n// generate parser-valid tokens, but if the active constrained content grows\n// beyond a local rollback limit before becoming complete, roll back only the\n// constrained suffix to the nearest valid non-dead prefix and continue from\n// that repaired state.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: Opening a span sets currentConstrainedOut := [], which is\n//   valid. Closing a complete span exits constrained mode. ConstrainedStep plus\n//   AppendConstrainedToken preserves validity on non-EOS tokens. The rollback\n//   branch uses RollbackConstrainedSuffix, whose postcondition gives a valid\n//   repaired currentConstrainedOut.\n// progress: UnconstrainedStep, CloseConstrainedSpan, and ConstrainedStep each\n//   consume one step and append at most one token. RollbackConstrainedSuffix\n//   shrinks or preserves generated and we still increment steps by 1, so the\n//   output-length bound remains true while the loop metric decreases.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar steps: nat := 0;\nvar rollbackLimit: nat := 24;\n\nwhile steps < maxSteps\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  decreases maxSteps - steps\n{{\n  if !insideConstrainedOut {{\n    var next := helpers.UnconstrainedStep(lm, prompt, generated);\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      generated := generated + [next];\n      if RenderedEndsWith(generated, "<<") {{\n        insideConstrainedOut := true;\n        currentConstrainedOut := [];\n      }}\n    }}\n  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{\n    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(\n      lm, parser, generated, currentConstrainedOut\n    );\n    generated := closedGenerated;\n    insideConstrainedOut := closedInside;\n    currentConstrainedOut := closedCurrent;\n    steps := steps + 1;\n  }} else if |currentConstrainedOut| >= rollbackLimit {{\n    var rolledGenerated, rolledCurrent := helpers.RollbackConstrainedSuffix(\n      parser, generated, currentConstrainedOut\n    );\n    generated := rolledGenerated;\n    insideConstrainedOut := true;\n    currentConstrainedOut := rolledCurrent;\n    steps := steps + 1;\n  }} else {{\n    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];\n    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(\n        lm, parser, generated, currentConstrainedOut, next\n      );\n      generated := appendedGenerated;\n      insideConstrainedOut := appendedInside;\n      currentConstrainedOut := appendedCurrent;\n    }}\n  }}\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Logit-shaped constrained CSD. The strategy uses ordinary free generation\n// outside spans. Inside a span, parser validity remains hard; the only extra\n// policy is a local soft preference: avoid closing very short constrained\n// prefixes (using caller-provided penalty tokens) and softly prefer operator\n// tokens once the prefix has begun (using caller-provided boost tokens). The\n// safe helpers filter literal token lists internally, so the strategy does not\n// need separate vocabulary-membership state for those lists.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: Outside the span the implication is vacuous unless a "<<"\n//   token is observed, in which case currentConstrainedOut becomes [].\n//   CloseConstrainedSpan exits constrained mode. Both safe constrained-step\n//   helpers return EOS or a token preserving parser validity, and\n//   AppendConstrainedToken carries that validity into currentConstrainedOut.\n// progress: UnconstrainedStep, CloseConstrainedSpan, and each safe constrained\n//   step consume one token-step and append at most one visible token, so\n//   |generated| <= |generatedPrefix| + steps is preserved.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar steps: nat := 0;\nvar minPrefixLength: nat := 2; // Minimum constrained prefix length before switching to boost mode\nvar penaltyTokens: seq<Token> := []; // Caller-provided tokens to penalize (e.g., early terminators like ">>")\nvar boostTokens: seq<Token> := []; // Caller-provided tokens to boost (e.g., operators)\n\nwhile steps < maxSteps\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  decreases maxSteps - steps\n{{\n  if !insideConstrainedOut {{\n    var next := helpers.UnconstrainedStep(lm, prompt, generated);\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      generated := generated + [next];\n      if RenderedEndsWith(generated, "<<") {{\n        insideConstrainedOut := true;\n        currentConstrainedOut := [];\n      }}\n    }}\n  }} else if parser.IsCompletePrefix(currentConstrainedOut) {{\n    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(\n      lm, parser, generated, currentConstrainedOut\n    );\n    generated := closedGenerated;\n    insideConstrainedOut := closedInside;\n    currentConstrainedOut := closedCurrent;\n    steps := steps + 1;\n  }} else {{\n    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];\n    var next := eosToken;\n    if |currentConstrainedOut| < minPrefixLength && |penaltyTokens| > 0 {{\n      next := helpers.SafePenalizedConstrainedStep(\n        lm, parser, constrainedPrompt, currentConstrainedOut, penaltyTokens, 6.0, eosToken\n      );\n    }} else if |boostTokens| > 0 {{\n      next := helpers.SafeBoostedConstrainedStep(\n        lm, parser, constrainedPrompt, currentConstrainedOut, boostTokens, 2.0, eosToken\n      );\n    }} else {{\n      next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);\n    }}\n    steps := steps + 1;\n    if next == eosToken {{\n      break;\n    }} else {{\n      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(\n        lm, parser, generated, currentConstrainedOut, next\n      );\n      generated := appendedGenerated;\n      insideConstrainedOut := appendedInside;\n      currentConstrainedOut := appendedCurrent;\n    }}\n  }}\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Grounded-and-closed constrained CSD. Phase 1 generates unconstrained until\n// "<<" opens a visible span, reserving step budget. Phase 2 fills the span with\n// RegenerateUnitOnGroundingFailure using HALF the remaining budget, so a reserve\n// is kept for closing; the helper rewinds and resamples any completed unit whose\n// identifier-like tokens are not grounded in the prompt context. Phase 3 calls\n// CloseSpanWithinBudget on the remaining budget to bring the span to a completable\n// state and emit the closing ">>". Per-phase budgets keep the length/cost bounds.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: Phase 1 sets insideConstrainedOut true only when next renders with suffix "<<",\n//   clearing currentConstrainedOut to [] (a valid prefix). Phase 2 stores the\n//   helper\'s result (parser-valid by its postcondition) into currentConstrainedOut.\n//   Phase 3 stores CloseSpanWithinBudget\'s result: on close, insideConstrainedOut is\n//   false and currentConstrainedOut == []; otherwise currentConstrainedOut stays a\n//   valid prefix. So !insideConstrainedOut ==> currentConstrainedOut == [] and\n//   insideConstrainedOut ==> IsValidPrefix(currentConstrainedOut) both hold.\n// length/cost: Phase 1 grows generated and steps by 1 per iteration, keeping\n//   |generated| <= |generatedPrefix| + steps. Phase 2 runs the helper with\n//   maxStepsPerUnit = fillBudget/4 (fillBudget = rem/2), maxRetries = 3; by its\n//   length/cost postconditions generated and cost grow by at most 4*(fillBudget/4)\n//   <= rem, added to steps. Phase 3 runs CloseSpanWithinBudget with closeBudget =\n//   maxSteps - steps; by its length/cost postconditions |generated| <=\n//   |generatedPrefix| + maxSteps, and we set steps := maxSteps.\n// progress: maxSteps > 0 ==> Phase 1 takes a step (cost > 0), or the span was open\n//   on entry and Phase 3 flips insideConstrainedOut.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar steps: nat := 0;\n\nwhile steps < maxSteps && !insideConstrainedOut\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  decreases maxSteps - steps\n{{\n  var next := helpers.UnconstrainedStep(lm, prompt, generated);\n  steps := steps + 1;\n  if next == eosToken {{\n    break;\n  }}\n  generated := generated + [next];\n  if RenderedEndsWith(generated, "<<") {{\n    insideConstrainedOut := true;\n    currentConstrainedOut := [];\n  }}\n}}\n\nif insideConstrainedOut && steps < maxSteps {{\n  var rem := maxSteps - steps;\n  var fillBudget := rem / 2;\n  if fillBudget >= 4 {{\n    var perUnit := fillBudget / 4;\n    assert fillBudget == 4 * (fillBudget / 4) + fillBudget % 4;\n    assert 4 * perUnit <= fillBudget;\n    assert 4 * perUnit <= rem;\n    var stable := generated[..|generated| - |currentConstrainedOut|];\n    var filled := helpers.RegenerateUnitOnGroundingFailure(\n      lm, parser, prompt + stable, currentConstrainedOut, eosToken, perUnit, 3, perUnit\n    );\n    generated := stable + filled;\n    currentConstrainedOut := filled;\n    steps := steps + 4 * perUnit;\n  }}\n}}\n\nif insideConstrainedOut && steps < maxSteps {{\n  var closeBudget := maxSteps - steps;\n  var cg, ci, cc := helpers.CloseSpanWithinBudget(\n    lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget\n  );\n  generated := cg;\n  insideConstrainedOut := ci;\n  currentConstrainedOut := cc;\n  steps := maxSteps;\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Open-then-reliably-close CSD. Phase 1 generates unconstrained until "<<" opens\n// a visible span. Phase 2 calls CloseSpanWithinBudget on the entire remaining\n// budget to advance the span to a completable state and emit the closing ">>",\n// leaving the span open only if no completable state is reachable within budget.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: Phase 1 flips insideConstrainedOut true only when next renders with suffix "<<",\n//   setting currentConstrainedOut := [] (valid). Phase 2\'s CloseSpanWithinBudget\n//   returns a closed span (insideConstrainedOut false, currentConstrainedOut [])\n//   or a still-open valid prefix, so both span invariants hold.\n// length/cost: Phase 1 grows generated and steps by 1 per iteration\n//   (|generated| <= |generatedPrefix| + steps). Phase 2 runs CloseSpanWithinBudget\n//   with closeBudget = maxSteps - steps; by its length/cost postconditions\n//   |generated| <= |generatedPrefix| + maxSteps, and we set steps := maxSteps.\n// progress: maxSteps > 0 ==> Phase 1 takes a step (cost > 0), or the span was open\n//   on entry and Phase 2 flips insideConstrainedOut.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar steps: nat := 0;\n\nwhile steps < maxSteps && !insideConstrainedOut\n  invariant 0 <= steps <= maxSteps\n  invariant lm.ValidTokensIdsLogits()\n  invariant !insideConstrainedOut ==> currentConstrainedOut == []\n  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)\n  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|\n  invariant |generated| <= |generatedPrefix| + steps\n  decreases maxSteps - steps\n{{\n  var next := helpers.UnconstrainedStep(lm, prompt, generated);\n  steps := steps + 1;\n  if next == eosToken {{\n    break;\n  }}\n  generated := generated + [next];\n  if RenderedEndsWith(generated, "<<") {{\n    insideConstrainedOut := true;\n    currentConstrainedOut := [];\n  }}\n}}\n\nif insideConstrainedOut && steps < maxSteps {{\n  var closeBudget := maxSteps - steps;\n  var cg, ci, cc := helpers.CloseSpanWithinBudget(\n    lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget\n  );\n  generated := cg;\n  insideConstrainedOut := ci;\n  currentConstrainedOut := cc;\n  steps := maxSteps;\n}}\n\ncost := steps;\n```\n\n```dafny\n// CSD_RATIONALE_BEGIN\n// Token-0 grounded constrained CSD. The eval surface is grammar-constrained from\n// the very first token and carries NO visible << >> delimiters, so this strategy\n// never emits "<<". If it is not already inside a constrained span it ENTERS one\n// silently via EnterObservedConstrainedSpan (sets inside=true, current=[], costs 0\n// tokens) instead of opening a visible span. It then GROUNDS the constrained\n// content unit-by-unit with RegenerateUnitOnGroundingFailure, which rewinds and\n// resamples any completed identifier whose tokens are unsupported by the prompt\n// context. Half the remaining step budget goes to grounding; the other half is\n// reserved so CloseSpanWithinBudget can always advance the span to a completable\n// state. Every budget is charged into `steps`, so the length and cost bounds hold.\n// CSD_RATIONALE_END\n// CSD_PROOF_SKETCH_BEGIN\n// parser_validity: EnterObservedConstrainedSpan returns current == [] (valid);\n//   RegenerateUnitOnGroundingFailure returns a valid prefix; CloseSpanWithinBudget\n//   returns either a closed span (current == []) or a valid open prefix. So both\n//   span invariants hold after every phase.\n// length/cost: steps starts at 0. The grounding phase spends fillBudget =\n//   (maxSteps - steps)/2 and raises steps by that amount; |generated| grows by at\n//   most fillBudget because the regenerated unit replaces a suffix of length\n//   |currentConstrainedOut| and RegenerateUnitOnGroundingFailure bounds the result\n//   by |currentConstrainedOut| + fillBudget. The close phase spends the rest\n//   (closeBudget = maxSteps - steps) and sets steps := maxSteps; by\n//   CloseSpanWithinBudget\'s |generatedOut| <= |generated| + closeBudget we get\n//   |generated| <= |generatedPrefix| + maxSteps. cost := steps <= maxSteps.\n// CSD_PROOF_SKETCH_END\ngenerated := generatedPrefix;\ninsideConstrainedOut := insideConstrained;\ncurrentConstrainedOut := currentConstrained;\ncost := 0;\n\nvar steps: nat := 0;\n\n// (1) Enter a constrained span with NO visible "<<" if not already inside.\nif !insideConstrainedOut {{\n  generated, insideConstrainedOut, currentConstrainedOut :=\n    helpers.EnterObservedConstrainedSpan(lm, generated);\n}}\nassert insideConstrainedOut;\nassert parser.IsValidPrefix(currentConstrainedOut);\nassert |currentConstrainedOut| <= |generated|;\nassert |generated| <= |generatedPrefix| + steps;\n\n// (2) Ground the constrained content unit-by-unit on half the remaining budget.\nif steps < maxSteps {{\n  var rem: nat := maxSteps - steps;\n  var fillBudget: nat := rem / 2;\n  if fillBudget >= 1 {{\n    assert |currentConstrainedOut| <= |generated|;\n    var stable := generated[..|generated| - |currentConstrainedOut|];\n    var filled := helpers.RegenerateUnitOnGroundingFailure(\n      lm, parser, prompt + stable, currentConstrainedOut, eosToken, fillBudget, 3, fillBudget);\n    generated := stable + filled;\n    currentConstrainedOut := filled;\n    steps := steps + fillBudget;\n    assert |generated| <= |generatedPrefix| + steps;\n  }}\n}}\nassert insideConstrainedOut;\nassert parser.IsValidPrefix(currentConstrainedOut);\nassert |currentConstrainedOut| <= |generated|;\nassert |generated| <= |generatedPrefix| + steps;\nassert steps <= maxSteps;\n\n// (3) Advance the span to a completable state on the reserved budget.\nif steps < maxSteps {{\n  var closeBudget: nat := maxSteps - steps;\n  generated, insideConstrainedOut, currentConstrainedOut :=\n    helpers.CloseSpanWithinBudget(\n      lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget);\n  steps := maxSteps;\n}}\nassert |generated| <= |generatedPrefix| + steps;\nassert steps <= maxSteps;\n\ncost := steps;\n```'

_VERIFIED_EXAMPLE_PREFIXES = (
    "// Guided-adaptive CSD.",
    "// Simple delimiter-triggered CSD.",
    "// Group-aware constrained CSD.",
    "// Adaptive-narrowness CSD.",
    "// Open-then-reliably-close CSD.",
)


def _build_allowed_helpers_block(allowed_helpers: list[str] | None) -> str:
    """Build a hard helper-call contract block for the current attempt."""
    if not allowed_helpers:
        return ""
    helper_names = ", ".join(f"`{name}`" for name in sorted(set(allowed_helpers)))
    return (
        "Helper-call contract for this attempt:\n"
        "Only these `helpers.<Method>(...)` and `CSDHelpers.<Method>(...)` calls are allowed:\n"
        f"{helper_names}\n"
        "Calls to helper or CSDHelpers methods outside this set are invalid for this attempt.\n\n"
    )


def _build_tool_reference_block(allowed_helpers: list[str] | None) -> str:
    """Build the helper/API reference the model sees for this attempt."""
    if not allowed_helpers:
        return _filter_tool_reference(sorted(_ALL_HELPER_NAMES)) + "\n\n"
    return _filter_tool_reference(allowed_helpers) + "\n\n"


def _build_verified_examples_block(allowed_helpers: list[str] | None) -> str:
    """Keep only verified examples compatible with the active helper contract."""
    allowed = set(allowed_helpers) if allowed_helpers is not None else None
    chunks = re.split(r"(?=// CSD_RATIONALE_BEGIN)", VERIFIED_EXAMPLES)
    kept_chunks = []
    for chunk in chunks:
        if "// CSD_RATIONALE_BEGIN" not in chunk:
            continue
        chunk = chunk.strip()
        if not any(prefix in chunk for prefix in _VERIFIED_EXAMPLE_PREFIXES):
            continue
        if allowed is not None:
            helper_names = set(_HELPER_CALL_RE.findall(chunk))
            if not helper_names.issubset(allowed):
                continue
            kept_chunks.append(chunk)
        else:
            kept_chunks.append(chunk)

    if kept_chunks:
        return "\n\n".join(kept_chunks).strip()
    return (
        "// No verified examples are compatible with the active helper-call "
        "contract for this attempt."
    )


def build_initial_prompt(
    task_description: str,
    allowed_helpers: list[str] | None = None,
) -> tuple[str, str]:
    model = InitialPromptModel(
        task_description=task_description,
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        verified_examples=_build_verified_examples_block(allowed_helpers),
    )
    user_prompt = _render_prompt(model, "author_prompts/initial_generation.j2")
    return SYSTEM_PROMPT, user_prompt


def build_verification_error_prompt(
    task_description: str,
    previous_strategy: str,
    error_message: str,
    behavioral_context: str = "",
    structured_feedback: str = "",
    error_history: str = "",
    strategy_context: str = "",
    search_memory: str = "",
    allowed_helpers: list[str] | None = None,
) -> tuple[str, str]:
    behavioral_context_block = ""
    structured_feedback_block = ""
    error_history_block = ""
    strategy_context_block = ""
    search_memory_block = ""
    if search_memory:
        search_memory_block = f"{search_memory}\n"
    if strategy_context:
        strategy_context_block = (
            "\nStrategy context from evaluated attempts before this verification failure:\n"
            f"{strategy_context}\n"
        )
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
    model = VerificationErrorPromptModel(
        task_description=task_description,
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        previous_strategy=previous_strategy,
        error_message=error_message,
        strategy_context_block=strategy_context_block,
        structured_feedback_block=structured_feedback_block,
        error_history_block=error_history_block,
        behavioral_context_block=behavioral_context_block,
        verified_examples=_build_verified_examples_block(allowed_helpers),
        search_memory_block=search_memory_block,
    )
    user_prompt = _render_prompt(model, "author_prompts/verification_error.j2")
    return SYSTEM_PROMPT, user_prompt


def build_runtime_error_prompt(
    previous_strategy: str,
    error_traceback: str,
    task_description: str = "Unknown task",
    search_memory: str = "",
    allowed_helpers: list[str] | None = None,
) -> tuple[str, str]:
    search_memory_block = f"{search_memory}\n" if search_memory else ""
    model = RuntimeErrorPromptModel(
        task_description=task_description,
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        previous_strategy=previous_strategy,
        error_traceback=error_traceback,
        search_memory_block=search_memory_block,
    )
    user_prompt = _render_prompt(model, "author_prompts/runtime_error.j2")
    return SYSTEM_PROMPT, user_prompt


def build_compilation_error_prompt(
    previous_strategy: str,
    error_message: str,
    search_memory: str = "",
    allowed_helpers: list[str] | None = None,
) -> tuple[str, str]:
    search_memory_block = f"{search_memory}\n" if search_memory else ""
    model = CompilationErrorPromptModel(
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        previous_strategy=previous_strategy,
        error_message=error_message,
        search_memory_block=search_memory_block,
    )
    user_prompt = _render_prompt(model, "author_prompts/compilation_error.j2")
    return SYSTEM_PROMPT, user_prompt


def build_format_repair_prompt(
    previous_strategy: str,
    search_memory: str = "",
    allowed_helpers: list[str] | None = None,
) -> tuple[str, str]:
    search_memory_block = f"{search_memory}\n" if search_memory else ""
    model = FormatRepairPromptModel(
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        previous_strategy=previous_strategy,
        search_memory_block=search_memory_block,
    )
    user_prompt = _render_prompt(model, "author_prompts/format_repair.j2")
    return SYSTEM_PROMPT, user_prompt


def _build_best_so_far_block(
    best_strategy: str | None,
    best_accuracy: float | None,
    best_syntax_rate: float | None,
) -> str:
    """Render the best-so-far strategy as a positive anchor.

    Shown in refinement mode only when the previous attempt is NOT the
    best-so-far — i.e. the previous attempt was a regression and the model
    should build from the better-scoring lineage instead. Empty otherwise,
    so the prompt stays minimal when there is only one strategy to discuss.
    """
    if (
        not best_strategy
        or best_accuracy is None
        or best_syntax_rate is None
    ):
        return ""
    return (
        "\n## Best result so far\n\n"
        f"```dafny\n{best_strategy}\n```\n\n"
        f"Result: accuracy {best_accuracy:.1%}, "
        f"syntax {best_syntax_rate:.1%}.\n"
        "The previous attempt regressed from this; consider building from "
        "this strategy instead.\n"
    )


def _build_eval_budget_block(eval_max_seconds_per_example: float | None) -> str:
    if eval_max_seconds_per_example is None:
        return ""
    return (
        f"Each example: {eval_max_seconds_per_example:.0f}s wall-clock budget; "
        "over-budget examples score 0.\n"
    )


def build_evaluation_failure_prompt(
    task_description: str,
    previous_strategy: str,
    previous_accuracy: float,
    previous_syntax_rate: float,
    num_examples: int,
    goal_accuracy: float,
    goal_syntax_rate: float,
    evaluation_feedback: str,
    best_strategy: str | None = None,
    best_accuracy: float | None = None,
    best_syntax_rate: float | None = None,
    search_memory: str = "",
    allowed_helpers: list[str] | None = None,
    eval_max_seconds_per_example: float | None = None,
    mode_examples: str = "",
    attempt_outcome_ledger: str = "",
) -> tuple[str, str]:
    search_memory_block = f"{search_memory}\n" if search_memory else ""
    best_so_far_block = _build_best_so_far_block(
        best_strategy=best_strategy,
        best_accuracy=best_accuracy,
        best_syntax_rate=best_syntax_rate,
    )
    mode_examples_block = (
        f"\n## Concrete failing rollouts from prior attempt\n\n{mode_examples}\n"
        if mode_examples
        else ""
    )
    attempt_outcome_ledger_block = (
        f"\n## Attempt outcome ledger\n\n{attempt_outcome_ledger}\n"
        if attempt_outcome_ledger
        else ""
    )
    accuracy_gap_pp = max(0.0, (goal_accuracy - previous_accuracy) * 100.0)
    syntax_gap_pp = max(0.0, (goal_syntax_rate - previous_syntax_rate) * 100.0)
    model = EvaluationFailurePromptModel(
        task_description=task_description,
        allowed_helpers_block=_build_allowed_helpers_block(allowed_helpers),
        tool_reference_block=_build_tool_reference_block(allowed_helpers),
        previous_strategy=previous_strategy,
        previous_accuracy_str=f"{previous_accuracy:.1%}",
        previous_syntax_rate_str=f"{previous_syntax_rate:.1%}",
        num_examples=num_examples,
        goal_accuracy_str=f"{goal_accuracy:.1%}",
        goal_syntax_rate_str=f"{goal_syntax_rate:.1%}",
        accuracy_gap_pp_str=f"{accuracy_gap_pp:.1f}",
        syntax_gap_pp_str=f"{syntax_gap_pp:.1f}",
        evaluation_feedback=evaluation_feedback,
        attempt_outcome_ledger_block=attempt_outcome_ledger_block,
        mode_examples_block=mode_examples_block,
        verified_examples=_build_verified_examples_block(allowed_helpers),
        search_memory_block=search_memory_block,
        best_so_far_block=best_so_far_block,
        eval_budget_block=_build_eval_budget_block(eval_max_seconds_per_example),
    )
    user_prompt = _render_prompt(model, "author_prompts/evaluation_failure.j2")
    return SYSTEM_PROMPT, user_prompt
