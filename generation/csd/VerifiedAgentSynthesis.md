# Verified Agent Synthesis Helper Surface

This file documents the public helper surface for generated CSD strategy bodies.

The library provides **orthogonal primitives** that strategies compose freely.
No step function hardcodes delimiter policy or assumes a particular strategy
shape (CRANE, IterGen, rollback-repair, etc.).  The strategy decides:

- **when** to emit delimiters (via `AppendLeftDelimiter` / `AppendRightDelimiter`)
- **how hard** to constrain (unconstrained → soft penalty → top-k → hard mask)
- **what local state** to maintain (counters, checkpoints, penalty schedules)

All answers must be wrapped in `<< ... >>` for evaluator extraction.

## Core Types

| Name | Meaning |
|------|---------|
| `Token` | `str` token emitted by the LM. |
| `Prefix` | `list[Token]`; the generated token prefix. |
| `LeftDelimiter` | `"<<"`; structural open token. |
| `RightDelimiter` | `">>"`; structural close token. |
| `SpacedLeftDelimiter` | `" <<"`; tokenizer variant. |
| `SpacedRightDelimiter` | `" >>"`; tokenizer variant. |

## Grammar State Queries

These route all parser queries through `LongestValidSuffix(generated)` so
strategies never need to call `parser.*` directly.

| Helper | Returns |
|--------|---------|
| `LongestValidSuffix(generated)` | Longest suffix of `generated` that is a valid parser prefix. |
| `CanConstrain(generated)` | `True` when the grammar suffix is incomplete (more tokens needed). |
| `IsComplete(generated)` | `True` when the grammar suffix is a complete parse. |
| `IsDead(generated)` | `True` when the grammar suffix cannot be extended or completed. |
| `ValidContinuationCount(generated)` | Number of grammar-valid next tokens. `1` = forced move, `0` = complete or dead. |
| `ParserDistanceToComplete(generated)` | Lower bound on tokens needed to reach a complete parse. |
| `MinStepsToComplete(generated)` | Alias for `ParserDistanceToComplete`. |

## Delimiter Predicates

Thin wrappers that handle both spaced and unspaced delimiter variants.

| Helper | Returns |
|--------|---------|
| `IsLeftDelimiterToken(token)` | `True` for `<<` and ` <<`. |
| `IsRightDelimiterToken(token)` | `True` for `>>` and ` >>`. |
| `EndsWithLeftDelimiter(generated)` | `True` when the last emitted token opened a span. |
| `EndsWithRightDelimiter(generated)` | `True` when the last emitted token closed a span. |
| `ContainsLeftDelimiter(generated)` | `True` when any left delimiter appears in `generated`. |
| `ContainsRightDelimiter(generated)` | `True` when any right delimiter appears in `generated`. |

## Primitive Step Functions

Each does **exactly one thing**: generate logits, apply one shaping policy,
choose.  All return `(nextToken, stepsLeft - 1)`.

| Step | Constraint | Use when |
|------|-----------|----------|
| `UnconstrainedStep(prompt, generated, stepsLeft)` | None | Free-form reasoning, any phase. |
| `ConstrainedStep(prompt, generated, stepsLeft)` | Hard grammar mask | Inside a constrained span, grammar must be enforced. Requires `CanConstrain(generated)`. |
| `SoftConstrainedStep(prompt, generated, penalty, stepsLeft)` | Grammar-invalid tokens biased by `-penalty` | Graduated constraint; LM can override grammar if confident enough. |
| `TopKConstrainedStep(prompt, generated, k, stepsLeft)` | Top-k filter then grammar mask | "Confident AND grammar-valid" selection. |
| `ForcedTokenStep(prompt, generated, token, stepsLeft)` | Returns `token` directly | Emitting delimiters, structural tokens, separators. |

**Key difference from old API:** `UnconstrainedStep` does not mask delimiters.
The strategy controls delimiter flow explicitly.  If you want to prevent
accidental delimiter emission during free-form text, call `MaskAllDelimiters`
after `GenerateLogits`, or use the logit shaping composites below.

## Logit Shaping Composites

Call these **after** `lm.GenerateLogits(...)` and **before**
`lm.ChooseNextToken()` to layer multiple shaping policies in one step.
They compose freely in any order.

| Helper | Effect |
|--------|--------|
| `SoftConstrainToGrammar(prefix, penalty)` | Bias grammar-invalid tokens by `-penalty`.  Grammar-valid tokens untouched. |
| `IntersectWithGrammar(prefix)` | Hard-mask everything not grammar-valid.  Grammar-valid tokens untouched. |
| `BiasForCompletion(prefix, bonus)` | Bias tokens that would complete the grammar by `+bonus`. |
| `MaskAllDelimiters(generated)` | Mask all four delimiter variants (`<<`, `>>`, ` <<`, ` >>`). |
| `MaskRightDelimiters(generated)` | Mask right delimiters only; left delimiters remain choosable. |
| `MaskLeftDelimiters(generated)` | Mask left delimiters only; right delimiters remain choosable. |
| `BiasLeftDelimiters(bias)` | Bias left delimiter variants by `+bias`. |
| `BiasRightDelimiters(bias)` | Bias right delimiter variants by `+bias`. |

### Example: Custom Step with Composed Shaping

```python
# Generate logits, soft-constrain to grammar, bias completion, mask right delimiters
lm.GenerateLogits(prompt + generated)
helpers.SoftConstrainToGrammar(generated, 10.0)
helpers.BiasForCompletion(generated, 3.0)
helpers.MaskRightDelimiters(generated)
next_token = lm.ChooseNextToken()
generated = generated + [next_token]
stepsLeft = stepsLeft - 1
```

## Append Wrappers

Convenience methods that call a step function and append the result.
Avoid stale-budget and forgotten-append mistakes.

| Helper | Wraps |
|--------|-------|
| `AppendUnconstrainedStep(prompt, prefix, stepsLeft)` | `UnconstrainedStep` |
| `AppendConstrainedStep(prompt, prefix, stepsLeft)` | `ConstrainedStep` |
| `AppendSoftConstrainedStep(prompt, prefix, penalty, stepsLeft)` | `SoftConstrainedStep` |
| `AppendTopKConstrainedStep(prompt, prefix, k, stepsLeft)` | `TopKConstrainedStep` |
| `AppendForcedToken(prefix, token, stepsLeft)` | `ForcedTokenStep` |
| `AppendLeftDelimiter(prefix, stepsLeft)` | `ForcedTokenStep` with `LeftDelimiter` |
| `AppendRightDelimiter(prefix, stepsLeft)` | `ForcedTokenStep` with `RightDelimiter` |

## Checkpoint Utilities

Lightweight local recovery without full rollback loops.

| Helper | Purpose |
|--------|---------|
| `Checkpoint(generated)` | Save a snapshot of the current prefix. |
| `RestoreCheckpoint(checkpoint)` | Restore exactly the saved prefix. |
| `RestoreIfDead(generated, checkpoint)` | Return `checkpoint` only when the grammar suffix is dead; otherwise keep `generated`. |

## Budget Utilities

| Helper | Purpose |
|--------|---------|
| `HasBudget(stepsLeft, needed)` | Pure predicate: `stepsLeft >= needed`. |
| `MinStepsToComplete(prefix)` | Lower bound on steps needed to complete the grammar suffix. |

## LM-Level Primitives

Available on `helpers.lm` when strategies need direct logit control beyond
the composites above.

| Method | Effect |
|--------|--------|
| `lm.GenerateLogits(input)` | Populate logits for next-token prediction on `input`. |
| `lm.ChooseNextToken()` | Return highest-logit unmasked token. |
| `lm.MaskToken(token)` | Set one token's logit to `-1e9`. |
| `lm.MaskTokens(tokens)` | Mask a list of tokens. |
| `lm.MaskTokensExcept(tokens)` | Mask everything except allowlist. |
| `lm.BiasToken(token, delta)` | Add `delta` to one token's logit (clamped to `[-1e9, 1e9]`). |
| `lm.BiasTokens(tokens, delta)` | Bias a list of tokens. |
| `lm.ScaleToken(token, factor)` | Multiply one token's logit by `factor` (clamped, `factor != 0`). |
| `lm.ScaleTokens(tokens, factor)` | Scale a list of tokens. |
| `lm.ClampLogits(low, high)` | Clip all logits to `[low, high]`. |
| `lm.TopKFilter(k)` | Mask everything except the `k` highest-logit tokens. |
| `lm.IsMasked(token)` | Check if a token is masked. |
| `lm.HasUnmaskedToken()` | Check if any token is still selectable. |

## Example Strategy Skeletons

### CRANE-like (delimiter-switched)

```python
# Free-form reasoning, then open a constrained span
while stepsLeft > 0 and not helpers.EndsWithLeftDelimiter(generated):
    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
# Constrained span
while stepsLeft > 0 and helpers.CanConstrain(generated):
    generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
# Close
generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
```

### Graduated constraint (novel)

```python
penalty = 1.0
while stepsLeft > 0 and not helpers.EndsWithLeftDelimiter(generated):
    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
while stepsLeft > 0 and helpers.CanConstrain(generated):
    generated, stepsLeft = helpers.AppendSoftConstrainedStep(prompt, generated, penalty, stepsLeft)
    penalty = penalty + 2.0  # increasing constraint pressure
generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
```

### Budget-aware with completion bias (novel)

```python
while stepsLeft > 0 and not helpers.EndsWithLeftDelimiter(generated):
    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
while stepsLeft > 0 and helpers.CanConstrain(generated):
    if helpers.HasBudget(stepsLeft, helpers.MinStepsToComplete(generated) + 2):
        generated, stepsLeft = helpers.AppendSoftConstrainedStep(prompt, generated, 5.0, stepsLeft)
    else:
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
```

### Multi-pass custom shaping (novel)

```python
while stepsLeft > 0 and not helpers.EndsWithLeftDelimiter(generated):
    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
while stepsLeft > 0 and helpers.CanConstrain(generated):
    lm.GenerateLogits(prompt + generated)
    helpers.SoftConstrainToGrammar(generated, 8.0)
    helpers.BiasForCompletion(generated, 3.0)
    helpers.MaskLeftDelimiters(generated)
    next_token = lm.ChooseNextToken()
    generated = generated + [next_token]
    stepsLeft = stepsLeft - 1
generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
```

### Checkpoint-based recovery (novel)

```python
while stepsLeft > 0 and not helpers.EndsWithLeftDelimiter(generated):
    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
checkpoint = helpers.Checkpoint(generated)
attempts = 0
while stepsLeft > 0 and helpers.CanConstrain(generated) and attempts < 3:
    generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
    if helpers.IsDead(generated):
        generated = helpers.RestoreCheckpoint(checkpoint)
        attempts = attempts + 1
generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
```

## Required Loop Invariants

Every decoding loop needs:

```python
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
```