# VerifiedAgentSynthesis API Reference

This document describes the complete public API exposed by
`generation/csd/VerifiedAgentSynthesis.py`.

The library supports two strategy architectures:

1. **Suffix-based**: The strategy works with a flat `generated` prefix and
   the library infers grammar state via `LongestValidSuffix`. Simple to use
   but the LM sees the full prefix including delimiters and freeform text
   during constrained generation.

2. **Split-prefix**: The strategy explicitly tracks `stablePrefix` (freeform
   reasoning before `<<`) and `currentConstrained` (answer tokens after `<<`)
   as separate state. The LM is fed `prompt + stablePrefix + currentConstrained`
   during constrained steps, giving it a cleaner view of the answer-so-far.
   This is how the highest-accuracy strategies work.

Both architectures compose from the same primitives. The split-prefix helpers
(`OpenConstrainedSpan`, `CloseConstrainedSpan`, `AppendConstrainedToken`,
`AdaptiveConstrainedStep`) are not compatibility shims — they are first-class
building blocks that enable the most expressive strategies.

---

## Data Types and Constants

| Name | Type / Value | Purpose |
|------|-------------|---------|
| `Token` | `str` | A single token emitted by the LM. |
| `Prefix` | `list[Token]` | An ordered sequence of tokens. |
| `Id` | `int` | Integer index into the vocabulary. |
| `Logit` | `float` | Log-probability weight for a token, clamped to `[-1e9, 1e9]`. |
| `MODULE_NAME` | `"VerifiedDecoderAgent"` | Dafny module name for transpilation. |
| `LeftDelimiter` | `"<<"` | Opens an answer span. |
| `RightDelimiter` | `">>"` | Closes an answer span. |
| `SpacedLeftDelimiter` | `" <<"` | Tokenizer variant of left delimiter. |
| `SpacedRightDelimiter` | `" >>"` | Tokenizer variant of right delimiter. |

## Specification Metadata

### `DafnySpec` (frozen dataclass)
Holds formal metadata attached to functions/methods: `kind`, `reads`,
`modifies`, `requires`, `ensures`, `decreases`, `axiom`, `extern`.

### `dafny_spec(**kwargs) -> decorator`
Decorator factory. Attaches a `DafnySpec` to the decorated callable as
`obj.__dafny_spec__`.

---

## Top-Level Utility Predicates

### `Contains(s: str, sub: str) -> bool`
String containment. Returns `sub in s`.

### `PrefixContains(p: Prefix, t: Token) -> bool`
Token membership. Returns whether `t` appears anywhere in `p`.

### `DelimitedAnswerValidForParser(parser: Parser, prefix: Prefix) -> bool`
Checks whether `prefix` contains a closed `<< ... >>` span whose extracted
content is a non-empty, parser-valid prefix. Used by the evaluator for answer
extraction.

---

## `LM` Class — Language Model Interface

Manages the token vocabulary, id mapping, and mutable logit array. All
logit-shaping methods modify the array in place.

### Construction and Invariants

#### `__init__() -> None`
Extern constructor. Initializes vocabulary with at least the delimiter tokens.
Ensures `ValidTokensIdsLogits()`.

#### `ValidTokensIdsLogits() -> bool`
Core invariant predicate. Checks: `Tokens`, `Ids`, `Logits` arrays have equal
nonzero length; ids are sequential from 0; tokens are unique; every token maps
to an id and vice versa; all logits are in `[-1e9, 1e9]`. Nearly every other
method requires this.

#### `ValidTokensIdsLogitsAlways() -> None`
Axiom lemma asserting that `ValidTokensIdsLogits()` holds unconditionally.
Called to re-establish the invariant at proof boundaries.

### Token / Id / Logit Conversions

#### `IdToToken(id: Id) -> Token`
Returns `Tokens[id]`. Requires `id in Ids`. Ensures result is in `Tokens` and
round-trips with `TokenToId`.

#### `TokenToId(token: Token) -> Id`
Returns the index of `token` in `Tokens` by delegating to
`TokenToIdRecursive(token, 0)`. Requires `token in Tokens`. Ensures result
is in `Ids` and round-trips with `IdToToken`.

#### `TokenToIdRecursive(token: Token, offset: int) -> Id`
Recursively scans from `offset` until `Tokens[offset] == token`. Decreases
on `|Tokens| - offset`.

#### `IdToLogit(id: Id) -> Logit`
Returns `Logits[id]`. Requires `id in Ids`.

#### `TokenToLogit(token: Token) -> Logit`
Composes `TokenToId` then `IdToLogit`. Requires `token in Tokens`.

#### `TokensToLogits(tokens: Prefix) -> list[Logit]`
Recursively maps a non-empty token list to logits.

#### `IdsToLogits(ids: list[Id]) -> list[Logit]`
Recursively maps a non-empty id list to logits.

### Hard Masking

These set logits to `-1e9`, making tokens effectively unselectable.

#### `MaskToken(token: Token) -> None`
Sets `Logits[TokenToId(token)] = -1e9`. All other logits unchanged.

#### `MaskTokens(tokens: Prefix) -> None`
Masks every token in the list. Tokens outside the list are unchanged.

#### `MaskTokensExcept(tokens: Prefix) -> None`
Masks all vocabulary tokens *except* the provided allowlist. Allowlisted
token logits are unchanged.

#### `IsMasked(token: Token) -> bool`
Returns whether `Logits[TokenToId(token)] == -1e9`.

#### `HasUnmaskedToken() -> bool`
Returns whether at least one token is not masked.

### Soft Logit Shaping

These modify logits without fully zeroing them.

#### `BiasToken(token: Token, delta: Logit) -> None`
Adds `delta` to the logit of `token`, clamping to `[-1e9, 1e9]`. All other
logits unchanged. This is the fundamental additive shaping primitive.

#### `BiasTokens(tokens: Prefix, delta: Logit) -> None`
Applies `BiasToken(t, delta)` for each `t` in `tokens`. Tokens outside the
list are unchanged.

#### `ScaleToken(token: Token, factor: Logit) -> None`
Multiplies the logit of `token` by `factor`, clamping to `[-1e9, 1e9]`.
Requires `factor != 0.0`. All other logits unchanged.

#### `ScaleTokens(tokens: Prefix, factor: Logit) -> None`
Applies `ScaleToken(t, factor)` for each `t` in `tokens`. Requires
`factor != 0.0`. Tokens outside the list are unchanged.

#### `ClampLogits(low: Logit, high: Logit) -> None`
Clips every logit to `[low, high]`. Requires `-1e9 <= low <= high <= 1e9`.
Logits already in range are unchanged. Useful for normalization after
multiple bias/scale operations.

### Filtering

#### `TopKFilter(k: int) -> None`
Masks all tokens except the `k` with highest logits. Requires `1 <= k <= |Tokens|`.
Ensures at least one token remains unmasked and that no previously-masked
token becomes unmasked. Extern.

### Generation

#### `GenerateLogits(input: Prefix) -> None`
Extern hook. Populates the logit array with the LM's next-token distribution
conditioned on `input`. Modifies `Logits`. Preserves `ValidTokensIdsLogits()`.

#### `ChooseNextToken() -> Token`
Returns the highest-logit unmasked token. Raises if all tokens are masked.
Extern. Ensures `token in Tokens` and `!IsMasked(token)`.

---

## `Parser` Class — Grammar Oracle Interface

All methods except `IsDeadPrefix` and `ValidNextToken` are abstract/extern.
The parser is stateless — all state is carried in the `prefix` argument.

#### `IsValidPrefix(prefix: Prefix) -> bool`
Returns whether `prefix` is a syntactically valid partial parse. Ensures
every proper prefix of a valid prefix is also valid. Extern.

#### `EmptyPrefixIsValid() -> None`
Axiom lemma asserting `IsValidPrefix([])`.

#### `IsCompletePrefix(prefix: Prefix) -> bool`
Returns whether `prefix` is a complete, finished parse. Ensures
`IsValidPrefix(prefix)`. Extern.

#### `IsDeadPrefix(prefix: Prefix) -> bool`
Returns `!IsCompletePrefix(prefix) && |ValidNextTokens(prefix)| == 0`.
A dead prefix cannot be extended or completed.

#### `ValidNextToken(prefix: Prefix, token: Token) -> bool`
Returns `token in ValidNextTokens(prefix)`. Requires `IsValidPrefix(prefix)`.

#### `ValidNextTokens(prefix: Prefix) -> Prefix`
Returns the set of tokens that can validly extend `prefix`. Requires
`IsValidPrefix(prefix)`. Ensures every returned token produces a valid
extension, and either the prefix is complete or the set is non-empty. Extern.

#### `ValidContinuationCount(prefix: Prefix) -> int`
Returns `|ValidNextTokens(prefix)|`. Requires `IsValidPrefix(prefix)`.

#### `ParserDistanceToComplete(prefix: Prefix) -> int`
Lower bound on tokens needed to reach a complete parse from `prefix`.
Returns 0 iff `IsCompletePrefix(prefix)`. Extern.

---

## `Delimiter` Class — Answer Extraction Support

Used by the evaluator to extract `<< ... >>` spans. Strategies do not
typically call `Delimiter` methods directly.

#### `__init__(left: Token, right: Token)`
Requires `left != right`.

#### `LastLeftDelimiterIndex(prefix: Prefix) -> int`
Index of the last occurrence of `Left` in `prefix`, or `|prefix|` if none.

#### `FirstRightDelimiterIndex(content: Prefix) -> int`
Index of the first occurrence of `Right` in `content`, or `|content|` if none.

#### `GetDelimitedContent(prefix: Prefix) -> Prefix`
Extracts tokens after the last `Left` and before the next `Right`.

#### `InsideDelimitedWindow(prefix: Prefix) -> bool`
Whether we are inside an open `Left` that has not yet been closed by `Right`.

#### Lemmas
- `NoFirstRightDelimiterIndexMeansNoRight(content)`
- `InsideDelimitedWindowNoRight(prefix)`
- `GetDelimitedContentAppend(prefix, next)`
- `AppendLeftEntersWindow(prefix)`
- `FirstRightDelimiterAppendRight(content)`
- `LastLeftDelimiterAppendNonLeft(prefix, tok)`
- `AppendRightExitsWindow(prefix)`

---

## `CSDHelpers` Class — Strategy Building Blocks

The central helper class. Composes `LM` and `Parser` into reusable step
functions, logit shapers, span-state managers, and recovery utilities.

### Construction

#### `__init__(lm: LM, parser: Parser)`
Stores references. Requires `lm.ValidTokensIdsLogits()`.

### Core Proof Lemmas

#### `AllValidNextTokensInLM(content: Prefix) -> None`
Axiom. Asserts that every token in `parser.ValidNextTokens(content)` is
present in `lm.Tokens`. Required before masking to grammar-valid tokens.

#### `ValidNextTokensInLMAfterStep(content: Prefix, next: Token) -> None`
Axiom. Carries the valid-next-tokens-in-LM property forward after appending
`next` to `content`.

---

### Suffix-Based Grammar Alignment

#### `LongestValidSuffix(prefix: Prefix) -> Prefix`
Returns the longest suffix of `prefix` such that `parser.IsValidPrefix(suffix)`.
Returns `[]` if `prefix` is empty (since `[]` is always valid). If `prefix`
itself is valid, returns `prefix` unchanged. The result is always a true
suffix: `result[i] == prefix[|prefix| - |result| + i]`. Decreases on `|prefix|`.

#### `LongestValidSuffixAppend(prefix: Prefix, next: Token) -> None`
Axiom lemma. If `next` is a valid continuation of `LongestValidSuffix(prefix)`,
then the longest valid suffix of `prefix + [next]` grows by at least 1.

#### `LongestValidSuffixIsValid(prefix: Prefix) -> None`
Lemma re-establishing `parser.IsValidPrefix(LongestValidSuffix(prefix))`.

#### `LongestValidSuffixNotDead(prefix: Prefix) -> None`
Lemma asserting the suffix is either complete or extensible.

---

### Grammar State Queries

These route parser queries through `LongestValidSuffix` so strategies never
need to call `parser.*` directly on the full generated prefix.

#### `CanConstrain(prefix: Prefix) -> bool`
Returns `!parser.IsCompletePrefix(LongestValidSuffix(prefix))`. True when
the grammar suffix still needs more tokens.

#### `IsComplete(prefix: Prefix) -> bool`
Returns `parser.IsCompletePrefix(LongestValidSuffix(prefix))`. True when
the grammar suffix is a complete parse.

#### `IsDead(prefix: Prefix) -> bool`
Returns `parser.IsDeadPrefix(LongestValidSuffix(prefix))`. True when the
grammar suffix cannot be extended or completed.

#### `ValidContinuationCount(prefix: Prefix) -> int`
Returns `parser.ValidContinuationCount(LongestValidSuffix(prefix))`.
Count of 1 means forced move; 0 means complete or dead.

#### `ParserDistanceToComplete(prefix: Prefix) -> int`
Lower bound on tokens needed to complete the grammar suffix.

#### `ValidTokenCount(prefix: Prefix) -> int`
Alias for `ValidContinuationCount`. Compatibility name.

---

### Delimiter Predicates

Thin wrappers that handle both spaced and unspaced delimiter variants.

#### `IsLeftDelimiterToken(token: Token) -> bool`
Returns `token == LeftDelimiter or token == SpacedLeftDelimiter`.

#### `IsRightDelimiterToken(token: Token) -> bool`
Returns `token == RightDelimiter or token == SpacedRightDelimiter`.

#### `EndsWithLeftDelimiter(prefix: Prefix) -> bool`
Returns whether the last token in `prefix` is a left delimiter variant.

#### `EndsWithRightDelimiter(prefix: Prefix) -> bool`
Returns whether the last token in `prefix` is a right delimiter variant.

#### `ContainsLeftDelimiter(prefix: Prefix) -> bool`
Returns whether any left delimiter variant appears in `prefix`.

#### `ContainsRightDelimiter(prefix: Prefix) -> bool`
Returns whether any right delimiter variant appears in `prefix`.

---

### Prefix Scanning Utilities

These enable context-dependent strategy decisions by inspecting the
generated prefix.

#### `LastTokenBefore(generated: Prefix, target: Token) -> tuple[Token, bool]` — NEW
Scans `generated` backwards for the last occurrence of `target`. If found
at position `index` where `index > 0`, returns `(generated[index - 1], True)`.
If `target` is not found, or appears only at position 0 (no preceding token),
returns `("", False)`.

This enables strategies to inspect what token preceded a structural marker.
For example, checking whether an arithmetic operator preceded the last `>>`
to decide whether to open a new constrained span.

Requires: `|generated| >= 0`.
Ensures: if `result[1]` is true, then `result[0]` is a token that appeared
in `generated` immediately before an occurrence of `target`.

#### `CountOccurrences(generated: Prefix, target: Token) -> int` — NEW
Returns the number of times `target` appears in `generated`. Useful for
tracking how many constrained spans have been opened/closed.

Requires: `|generated| >= 0`.
Ensures: `result >= 0`.

#### `TokensSinceLastDelimiter(generated: Prefix) -> int` — NEW
Returns the number of tokens emitted since the last left or right delimiter
variant (any of `<<`, `>>`, ` <<`, ` >>`) in `generated`. Returns
`|generated|` if no delimiter has been emitted.

Useful for minimum-span-length policies (e.g., "don't close the span until
at least N constrained tokens have been emitted").

Requires: `|generated| >= 0`.
Ensures: `0 <= result <= |generated|`.

---

### Primitive Step Functions

Each does one thing: generate logits, apply one shaping policy, choose.
All return `(nextToken, stepsLeft - 1)`.

#### `UnconstrainedStep(prompt, generated, stepsLeft) -> (Token, int)`
Generates logits for `prompt + generated`, masks all four delimiter variants
(to prevent accidental span open/close during freeform reasoning), then
chooses. The baseline freeform step.

#### `UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft) -> (Token, int)`
Like `UnconstrainedStep` but only masks right delimiter variants. Left
delimiters remain choosable, so the LM can naturally decide to open a span.

#### `UnconstrainedBiasLeftDelimiterStep(prompt, generated, bias, stepsLeft) -> (Token, int)`
Like `UnconstrainedAllowLeftDelimiterStep` but additionally biases left
delimiter variants by `+bias`. Requires `bias > 0.0`. Use when the strategy
wants to encourage span opening.

#### `UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft) -> (Token, int)`
Convenience wrapper calling `UnconstrainedBiasLeftDelimiterStep` with a
built-in bias of `5.0`.

#### `ConstrainedStep(prompt, generated, stepsLeft) -> (Token, int)`
Computes `LongestValidSuffix(generated)`, masks all tokens except
`parser.ValidNextTokens(suffix)`, then chooses. Requires `CanConstrain(generated)`.
Ensures the chosen token is grammar-valid and extends the suffix.

#### `ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft) -> (Token, int)`
Like `ConstrainedStep` but when `IsComplete(generated)` is true, also allows
right delimiter variants. This lets the LM naturally choose to close a span
when the grammar is satisfied. Ensures: if the chosen token is a right
delimiter, then the grammar suffix was complete.

#### `SoftConstrainedStep(prompt, generated, penalty, stepsLeft) -> (Token, int)`
Generates logits, then biases grammar-invalid tokens by `-penalty` instead of
masking them. The LM can still select an invalid token if its logit is high
enough to overcome the penalty. Requires `penalty > 0.0`. Grammar-valid token
logits are untouched.

#### `TopKConstrainedStep(prompt, generated, k, stepsLeft) -> (Token, int)`
Generates logits, applies `TopKFilter(k)`, then intersects with grammar-valid
tokens. "Confident AND grammar-valid" selection. Requires `1 <= k <= |lm.Tokens|`.

#### `ForcedTokenStep(prompt, generated, token, stepsLeft) -> (Token, int)`
Returns `token` directly without consulting the LM. Requires `token in lm.Tokens`.
Used for emitting delimiters, separators, or structural tokens.

---

### Logit Shaping Composites

Call these after `lm.GenerateLogits(...)` and before `lm.ChooseNextToken()`
to layer multiple shaping policies in one step. They compose freely in any
order.

#### `SoftConstrainToGrammar(prefix: Prefix, penalty: Logit) -> None`
Computes `LongestValidSuffix(prefix)`, identifies grammar-invalid tokens,
biases them by `-penalty`. Grammar-valid token logits are untouched.

#### `IntersectWithGrammar(prefix: Prefix) -> None`
Computes `LongestValidSuffix(prefix)`, hard-masks everything not grammar-valid.
Grammar-valid token logits are untouched.

#### `BiasForCompletion(prefix: Prefix, bonus: Logit) -> None`
For each grammar-valid next token that would make the suffix complete, biases
it by `+bonus`. Tokens that don't complete the grammar are untouched. Requires
`bonus > 0.0`.

#### `MaskAllDelimiters(generated: Prefix) -> None`
Masks all four delimiter variants (`<<`, `>>`, ` <<`, ` >>`).

#### `MaskRightDelimiters(generated: Prefix) -> None`
Masks right delimiter variants only. Left delimiters remain choosable.

#### `MaskLeftDelimiters(generated: Prefix) -> None`
Masks left delimiter variants only. Right delimiters remain choosable.

#### `BiasLeftDelimiters(bias: Logit) -> None`
Biases left delimiter variants by `+bias`. Requires `bias > 0.0`.

#### `BiasRightDelimiters(bias: Logit) -> None`
Biases right delimiter variants by `+bias`. Requires `bias > 0.0`.

---

### Append Wrappers

Convenience methods that call a step function and append the result to the
prefix. Prevent stale-budget and forgotten-append mistakes.

| Wrapper | Wraps |
|---------|-------|
| `AppendUnconstrainedStep(prompt, prefix, stepsLeft)` | `UnconstrainedStep` |
| `AppendUnconstrainedAllowLeftDelimiterStep(prompt, prefix, stepsLeft)` | `UnconstrainedAllowLeftDelimiterStep` |
| `AppendUnconstrainedNudgeLeftDelimiterStep(prompt, prefix, stepsLeft)` | `UnconstrainedNudgeLeftDelimiterStep` |
| `AppendConstrainedStep(prompt, prefix, stepsLeft)` | `ConstrainedStep` |
| `AppendConstrainedOrRightDelimiterStep(prompt, prefix, stepsLeft)` | `ConstrainedOrRightDelimiterStep` |
| `AppendSoftConstrainedStep(prompt, prefix, penalty, stepsLeft)` | `SoftConstrainedStep` |
| `AppendTopKConstrainedStep(prompt, prefix, k, stepsLeft)` | `TopKConstrainedStep` |
| `AppendForcedToken(prefix, token, stepsLeft)` | `ForcedTokenStep` |
| `AppendLeftDelimiter(prefix, stepsLeft)` | `ForcedTokenStep` with `LeftDelimiter` |
| `AppendRightDelimiter(prefix, stepsLeft)` | `ForcedTokenStep` with `RightDelimiter` |

All return `(updatedPrefix, remainingSteps)` with `|updated| == |prefix| + 1`
and `remainingSteps == stepsLeft - 1`.

---

### Split-Prefix Span-State Helpers

These are the building blocks for strategies that explicitly track the
constrained answer content separately from freeform reasoning. They manage
the `(generated, insideSpan, currentConstrained)` state triple that the
highest-accuracy strategies use.

The key advantage: during constrained generation, the LM is fed
`prompt + stablePrefix + currentConstrained` rather than `prompt + generated`.
This gives the LM a cleaner view of the answer-so-far without delimiter tokens
and interleaved freeform text in the way.

#### `OpenConstrainedSpan(prefix, stepsLeft) -> (Prefix, bool, Prefix, int)`
Appends `LeftDelimiter` to `prefix` and returns:
- `updated`: `prefix + [LeftDelimiter]`
- `insideSpan`: `True`
- `currentConstrained`: `[]` (empty — the span just opened)
- `remainingSteps`: `stepsLeft - 1`

Requires `LeftDelimiter in lm.Tokens` and `stepsLeft >= 1`.

#### `CloseConstrainedSpan(prefix, currentConstrained, stepsLeft) -> (Prefix, bool, Prefix, int)`
Appends `RightDelimiter` to `prefix` and returns:
- `updated`: `prefix + [RightDelimiter]`
- `insideSpan`: `False`
- `currentConstrained`: `[]` (reset — the span is closed)
- `remainingSteps`: `stepsLeft - 1`

Requires `parser.IsCompletePrefix(currentConstrained)` — the span content
must be a complete parse before closing. Also requires the suffix invariant:
`prefix[|prefix| - |currentConstrained|..] == currentConstrained`.

#### `AppendConstrainedToken(prefix, currentConstrained, token) -> (Prefix, bool, Prefix)`
Appends `token` to both `prefix` and `currentConstrained`. Returns:
- `updated`: `prefix + [token]`
- `insideSpan`: `True`
- `updatedConstrained`: `currentConstrained + [token]`

Does NOT consume a step — it is a pure state update. The step was already
consumed by whichever step function chose `token`. Requires
`parser.ValidNextToken(currentConstrained, token)`. Ensures
`parser.IsValidPrefix(currentConstrained + [token])`.

#### `AdaptiveConstrainedStep(prompt, stablePrefix, currentConstrained, validTokenGroups, bonus, narrowThreshold, eosToken, stepsLeft) -> (Token, int)` — NEW
The core constrained step for split-prefix strategies.

**Behavior:**
1. Generates logits for `prompt + stablePrefix + currentConstrained` — the LM
   sees freeform reasoning as context, then the clean answer-so-far.
2. Masks to `parser.ValidNextTokens(currentConstrained)`.
3. If `parser.ValidContinuationCount(currentConstrained) > narrowThreshold`,
   biases tokens appearing in any group in `validTokenGroups` by `+bonus`.
   This encourages domain-relevant tokens when the grammar is wide open but
   has no effect in bottleneck states where grammar alone suffices.
4. Chooses the next token.
5. If the chosen token equals `eosToken`, returns `(eosToken, stepsLeft - 1)`.
6. Otherwise returns `(nextToken, stepsLeft - 1)`.

**Parameters:**
- `prompt`: the task prompt
- `stablePrefix`: everything before the current span's content (`generated[..|generated| - |currentConstrained|]`)
- `currentConstrained`: the answer tokens so far (after `<<`)
- `validTokenGroups`: `list[list[Token]]` — groups of domain-relevant tokens to boost (e.g., digit tokens, operator tokens)
- `bonus`: positive logit bias for tokens in `validTokenGroups`
- `narrowThreshold`: only apply group boost when valid token count exceeds this
- `eosToken`: end-of-sequence token; if chosen, signals early termination
- `stepsLeft`: remaining budget

Requires: `parser.IsValidPrefix(currentConstrained)`,
`!parser.IsCompletePrefix(currentConstrained)`, `stepsLeft >= 1`,
`bonus > 0.0`, and all tokens in `validTokenGroups` must be in `lm.Tokens`.

#### `GroupBoostedConstrainedStep(prompt, stablePrefix, currentConstrained, validTokenGroups, bonus, stepsLeft) -> (Token, int)`
Simplified `AdaptiveConstrainedStep` without `narrowThreshold` or `eosToken`.
Always applies the group boost when valid token groups are provided.

Requires: same as `AdaptiveConstrainedStep` minus `narrowThreshold`/`eosToken`.

#### `PenalizedConstrainedStep(prompt, stablePrefix, currentConstrained, penaltyTokens, penalty, stepsLeft) -> (Token, int)`
Split-prefix constrained step that additionally penalizes specific tokens by
`-penalty` within the grammar-valid set. Useful for discouraging particular
token choices while still enforcing grammar validity.

Requires: `parser.IsValidPrefix(currentConstrained)`,
`!parser.IsCompletePrefix(currentConstrained)`, `penalty > 0.0`,
all `penaltyTokens` in `lm.Tokens`.

---

### Checkpoint and Recovery Utilities

#### `Checkpoint(prefix: Prefix) -> Prefix`
Returns `prefix` unchanged. Semantically: "save a snapshot." The strategy
stores the return value in a local variable.

#### `RestoreCheckpoint(checkpoint: Prefix) -> Prefix`
Returns `checkpoint` unchanged. Semantically: "restore the saved snapshot."

#### `RestoreIfDead(prefix, checkpoint) -> Prefix`
Returns `checkpoint` if `IsDead(prefix)`, otherwise returns `prefix`.

---

### Budget Utilities

#### `HasBudget(stepsLeft: int, needed: int) -> bool`
Returns `stepsLeft >= needed`. Pure predicate.

#### `MinStepsToComplete(prefix: Prefix) -> int`
Returns `ParserDistanceToComplete(prefix)`. Lower bound on steps needed to
complete the grammar suffix.

---

## Exported Public Surface (`__all__`)

```python
[
    "MODULE_NAME",
    "Token", "Prefix", "Id", "Logit",
    "DafnySpec", "dafny_spec",
    "Contains", "PrefixContains", "DelimitedAnswerValidForParser",
    "LeftDelimiter", "RightDelimiter",
    "SpacedLeftDelimiter", "SpacedRightDelimiter",
    "LM", "Parser", "Delimiter", "CSDHelpers",
]
```