# Verified Agent Synthesis Helper Surface

This file documents the public helper surface intended for generated CSD
strategy bodies. The implementation lives in
`generation/csd/VerifiedAgentSynthesis.py`.

The library still contains low-level `LM`, `Parser`, and `Delimiter` machinery,
but generated strategies should use `CSDHelpers` wrappers instead of directly
calling parser or logit-shaping methods when a wrapper exists.

## Core Types

| Name | Meaning |
|------|---------|
| `Token` | `str` token emitted by the LM. |
| `Prefix` | `list[Token]`; the generated token prefix. |
| `LeftDelimiter` | `"<<";` structural open token. |
| `RightDelimiter` | `">>";` structural close token. |
| `SpacedLeftDelimiter` | `" <<";` tokenizer variant. |
| `SpacedRightDelimiter` | `" >>";` tokenizer variant. |

`generated` is a single token list containing free-form text, delimiters, and
constrained content. The evaluator treats delimiter spans as parseable islands;
the final `<< ... >>` span is the default graded answer.

## Strategy-Facing Helpers

### Natural Free-Form Steps

| Helper | Purpose |
|--------|---------|
| `AppendUnconstrainedStep(prompt, generated, stepsLeft)` | Appends one free-form token while masking delimiter tokens. Use for ordinary reasoning. |
| `AppendUnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)` | Appends one free-form token while allowing `<<` / ` <<`. |
| `AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)` | Appends one free-form token while biasing `<<` / ` <<`. Use once a state policy says the answer span should open soon. |
| `UnconstrainedStep(prompt, generated, stepsLeft)` | Raw token-returning version of the ordinary free-form step. |
| `UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)` | Raw token-returning version of the allow-left-delimiter step. |
| `UnconstrainedBiasLeftDelimiterStep(prompt, generated, bias, stepsLeft)` | Raw token-returning left-delimiter step with caller-provided positive bias. |
| `UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)` | Raw token-returning left-delimiter step with built-in positive bias. |

Prefer the append-style helpers in generated strategies. They avoid stale
budget and forgotten-append mistakes.

### Constrained Span Steps

| Helper | Purpose |
|--------|---------|
| `AppendConstrainedStep(prompt, generated, stepsLeft)` | Appends one grammar-valid token while `helpers.CanConstrain(generated)` is true. |
| `AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)` | Appends a grammar-valid continuation, or `>>` / ` >>` only when `helpers.IsComplete(generated)` is true. |
| `ConstrainedStep(prompt, generated, stepsLeft)` | Raw token-returning hard grammar step. |
| `ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)` | Raw token-returning grammar-or-close step. |

`AppendConstrainedOrRightDelimiterStep` is the preferred natural-close helper:
it lets the LM choose the close token, but only after parser completion.

### Delimiter Predicates

| Helper | Purpose |
|--------|---------|
| `IsLeftDelimiterToken(token)` | True for `LeftDelimiter` and `SpacedLeftDelimiter`. |
| `IsRightDelimiterToken(token)` | True for `RightDelimiter` and `SpacedRightDelimiter`. |
| `EndsWithLeftDelimiter(generated)` | True when the latest emitted token opened a span. |
| `EndsWithRightDelimiter(generated)` | True when the latest emitted token closed a span. |

These wrappers keep strategies from forgetting spaced delimiter variants.

### Parser-State Wrappers

| Helper | Purpose |
|--------|---------|
| `LongestValidSuffix(generated)` | Longest suffix of `generated` accepted as a parser-valid prefix. |
| `CanConstrain(generated)` | True when the current grammar suffix is incomplete. |
| `IsComplete(generated)` | True when the current grammar suffix is complete. |
| `IsDead(generated)` | True when the current grammar suffix cannot be completed. |
| `ValidContinuationCount(generated)` | Number of valid next tokens from the current grammar suffix. |
| `ParserDistanceToComplete(generated)` | Lower bound on steps needed to complete the current grammar suffix. |
| `MinStepsToComplete(generated)` | Alias for `ParserDistanceToComplete(generated)`. |
| `HasBudget(stepsLeft, needed)` | Pure budget predicate. |

Generated strategies should prefer these helpers over direct `parser.*` calls.
They automatically route parser queries through `LongestValidSuffix(generated)`.

### Explicit Structural Tokens

These remain available for non-natural delimiter strategies and non-GSM tasks.
GSM natural-delimiter runs should not use them for delimiters.

| Helper | Purpose |
|--------|---------|
| `ForcedTokenStep(prompt, generated, token, stepsLeft)` | Raw token-returning forced structural token step. |
| `AppendForcedToken(generated, token, stepsLeft)` | Append a known token. |
| `AppendLeftDelimiter(generated, stepsLeft)` | Append `LeftDelimiter`. |
| `AppendRightDelimiter(generated, stepsLeft)` | Append `RightDelimiter`. |

## Removed From The Strategy Surface

The following experimental helpers were removed from `CSDHelpers` because they
were distracting the synthesis model or weakening the final-span guarantee:

- soft constrained decoding helpers
- top-k constrained decoding helpers
- budget-aware switching helpers
- extend-constrained helpers
- rollback/salvage/retry helpers
- checkpoint and repetition state structures
- direct composite logit-shaping helpers

The intended GSM natural path is now:

1. Reason with `AppendUnconstrainedStep`.
2. When a policy believes the final answer is near, use
   `AppendUnconstrainedNudgeLeftDelimiterStep`.
3. Once `EndsWithLeftDelimiter(generated)` is true, use
   `AppendConstrainedOrRightDelimiterStep`.
4. Once `EndsWithRightDelimiter(generated)` is true, either stop if this was
   the final answer span or return to free-form reasoning before a later final
   span.

Every decoding loop still needs the standard invariants:

```python
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
```
