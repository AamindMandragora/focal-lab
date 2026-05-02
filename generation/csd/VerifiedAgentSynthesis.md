# VerifiedAgentSynthesis API Reference (Standalone)

This document describes the complete public API exposed by
`generation/csd/VerifiedAgentSynthesis.py`, organized by logical subsystem.

## Data Types And Constants

- `Token = str`
- `Prefix = list[Token]`
- `Id = int`
- `Logit = float`
- `MODULE_NAME = "VerifiedDecoderAgent"`
- `LeftDelimiter = "<<"`
- `RightDelimiter = ">>"`
- `SpacedLeftDelimiter = " <<"`
- `SpacedRightDelimiter = " >>"`

## Specification Metadata

### `DafnySpec` (dataclass)
- Holds formal metadata attached to functions/methods:
  `kind`, `reads`, `modifies`, `requires`, `ensures`, `decreases`, `axiom`, `extern`.

### `dafny_spec(...)`
- Decorator factory that attaches a `DafnySpec` object to decorated callables.

## Top-Level Utility Predicates

### `Contains(s: str, sub: str) -> bool`
- String containment predicate.

### `PrefixContains(p: Prefix, t: Token) -> bool`
- Token containment predicate over a prefix.

### `DelimitedAnswerValidForParser(parser: Parser, prefix: Prefix) -> bool`
- Checks whether `prefix` contains a balanced delimited segment and that extracted content is parser-valid.

## `LM` Class (Language Model Wrapper)

### Construction And Invariants
- `__init__()`
- `ValidTokensIdsLogits() -> bool`
- `ValidTokensIdsLogitsAlways() -> None`

### Id/Token/Logit Mapping
- `IdToToken(id: Id) -> Token`
- `TokenToId(token: Token) -> Id`
- `TokenToIdRecursive(token: Token, offset: int) -> Id`
- `IdToLogit(id: Id) -> Logit`
- `TokenToLogit(token: Token) -> Logit`
- `TokensToLogits(tokens: Prefix) -> list[Logit]`
- `IdsToLogits(ids: list[Id]) -> list[Logit]`

### Hard Masking
- `MaskToken(token: Token) -> None`
- `MaskTokens(tokens: Prefix) -> None`
- `MaskTokensExcept(tokens: Prefix) -> None`
- `IsMasked(token: Token) -> bool`
- `HasUnmaskedToken() -> bool`

### Logit Shaping
- `BiasToken(token: Token, delta: Logit) -> None`
- `BiasTokens(tokens: Prefix, delta: Logit) -> None`
- `ScaleToken(token: Token, factor: Logit) -> None`
- `ScaleTokens(tokens: Prefix, factor: Logit) -> None`
- `ClampLogits(low: Logit, high: Logit) -> None`

### Filtering And Sampling
- `TopKFilter(k: int) -> None`
- `GenerateLogits(input: Prefix) -> None`
- `ChooseNextToken() -> Token`

## `Parser` Class (Grammar Oracle)

### Prefix Validity And Completeness
- `IsValidPrefix(prefix: Prefix) -> bool`
- `EmptyPrefixIsValid() -> None`
- `IsCompletePrefix(prefix: Prefix) -> bool`
- `IsDeadPrefix(prefix: Prefix) -> bool`

### Next-Token Semantics
- `ValidNextToken(prefix: Prefix, token: Token) -> bool`
- `ValidNextTokens(prefix: Prefix) -> Prefix`
- `ValidContinuationCount(prefix: Prefix) -> int`
- `ParserDistanceToComplete(prefix: Prefix) -> int`

## `Delimiter` Class (Delimited-Window Utilities)

### Construction
- `__init__(left: Token, right: Token)`

### Window Indices
- `LastLeftDelimiterIndex(prefix: Prefix) -> int`
- `FirstRightDelimiterIndex(content: Prefix) -> int`

### Content Extraction
- `GetDelimitedContent(prefix: Prefix) -> Prefix`
- `InsideDelimitedWindow(prefix: Prefix) -> bool`

### Delimiter Lemmas And Transition Helpers
- `NoFirstRightDelimiterIndexMeansNoRight(content: Prefix) -> None`
- `InsideDelimitedWindowNoRight(prefix: Prefix) -> None`
- `GetDelimitedContentAppend(prefix: Prefix, next: Token) -> None`
- `AppendLeftEntersWindow(prefix: Prefix) -> None`
- `FirstRightDelimiterAppendRight(content: Prefix) -> None`
- `LastLeftDelimiterAppendNonLeft(prefix: Prefix, tok: Token) -> None`
- `AppendRightExitsWindow(prefix: Prefix) -> None`

## `CSDHelpers` Class (Strategy Building Blocks)

### Construction
- `__init__(lm: LM, parser: Parser)`

### Core Proof/Consistency Lemmas
- `AllValidNextTokensInLM(content: Prefix) -> None`
- `ValidNextTokensInLMAfterStep(content: Prefix, next: Token) -> None`

### Suffix Alignment
- `LongestValidSuffix(prefix: Prefix) -> Prefix`
- `LongestValidSuffixAppend(prefix: Prefix, next: Token) -> None`
- `LongestValidSuffixIsValid(prefix: Prefix) -> None`
- `LongestValidSuffixNotDead(prefix: Prefix) -> None`

### Grammar State Wrappers
- `CanConstrain(prefix: Prefix) -> bool`
- `IsComplete(prefix: Prefix) -> bool`
- `IsDead(prefix: Prefix) -> bool`
- `ValidContinuationCount(prefix: Prefix) -> int`
- `ParserDistanceToComplete(prefix: Prefix) -> int`

### Delimiter Predicates
- `IsLeftDelimiterToken(token: Token) -> bool`
- `IsRightDelimiterToken(token: Token) -> bool`
- `EndsWithLeftDelimiter(prefix: Prefix) -> bool`
- `EndsWithRightDelimiter(prefix: Prefix) -> bool`
- `ContainsLeftDelimiter(prefix: Prefix) -> bool`
- `ContainsRightDelimiter(prefix: Prefix) -> bool`

### Primitive Step Methods (Token + Budget Delta)
- `UnconstrainedStep(prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]`
- `UnconstrainedAllowLeftDelimiterStep(prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]`
- `UnconstrainedBiasLeftDelimiterStep(prompt: Prefix, generated: Prefix, bias: Logit, stepsLeft: int) -> tuple[Token, int]`
- `UnconstrainedNudgeLeftDelimiterStep(prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]`
- `ConstrainedStep(prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]`
- `ConstrainedOrRightDelimiterStep(prompt: Prefix, generated: Prefix, stepsLeft: int) -> tuple[Token, int]`
- `SoftConstrainedStep(prompt: Prefix, generated: Prefix, penalty: Logit, stepsLeft: int) -> tuple[Token, int]`
- `TopKConstrainedStep(prompt: Prefix, generated: Prefix, k: int, stepsLeft: int) -> tuple[Token, int]`
- `ForcedTokenStep(prompt: Prefix, generated: Prefix, token: Token, stepsLeft: int) -> tuple[Token, int]`

### Logit-Shaping Composites
- `SoftConstrainToGrammar(prefix: Prefix, penalty: Logit) -> None`
- `IntersectWithGrammar(prefix: Prefix) -> None`
- `BiasForCompletion(prefix: Prefix, bonus: Logit) -> None`
- `MaskAllDelimiters(generated: Prefix) -> None`
- `MaskRightDelimiters(generated: Prefix) -> None`
- `BiasLeftDelimiters(bias: Logit) -> None`
- `BiasRightDelimiters(bias: Logit) -> None`
- `MaskLeftDelimiters(generated: Prefix) -> None`

### Append Wrappers (Prefix + Budget Delta)
- `AppendUnconstrainedStep(prompt: Prefix, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]`
- `AppendUnconstrainedAllowLeftDelimiterStep(prompt: Prefix, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]`
- `AppendUnconstrainedNudgeLeftDelimiterStep(prompt: Prefix, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]`
- `AppendConstrainedStep(prompt: Prefix, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]`
- `AppendConstrainedOrRightDelimiterStep(prompt: Prefix, prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]`
- `AppendSoftConstrainedStep(prompt: Prefix, prefix: Prefix, penalty: Logit, stepsLeft: int) -> tuple[Prefix, int]`
- `AppendTopKConstrainedStep(prompt: Prefix, prefix: Prefix, k: int, stepsLeft: int) -> tuple[Prefix, int]`
- `AppendForcedToken(prefix: Prefix, token: Token, stepsLeft: int) -> tuple[Prefix, int]`
- `AppendLeftDelimiter(prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]`
- `AppendRightDelimiter(prefix: Prefix, stepsLeft: int) -> tuple[Prefix, int]`

### Span-State Compatibility Helpers
- `ValidTokenCount(prefix: Prefix) -> int`
- `OpenConstrainedSpan(prefix: Prefix, stepsLeft: int) -> tuple[Prefix, bool, Prefix, int]`
- `CloseConstrainedSpan(prefix: Prefix, currentConstrained: Prefix, stepsLeft: int) -> tuple[Prefix, bool, Prefix, int]`
- `AppendConstrainedToken(prefix: Prefix, currentConstrained: Prefix, token: Token) -> tuple[Prefix, bool, Prefix]`
- `GroupBoostedConstrainedStep(prompt: Prefix, stablePrefix: Prefix, currentConstrained: Prefix, validTokenGroups: list[list[Token]], bonus: Logit, stepsLeft: int) -> tuple[Token, int]`
- `PenalizedConstrainedStep(prompt: Prefix, stablePrefix: Prefix, currentConstrained: Prefix, penaltyTokens: Prefix, penalty: Logit, stepsLeft: int) -> tuple[Token, int]`

### Recovery And Budget Utilities
- `Checkpoint(prefix: Prefix) -> Prefix`
- `RestoreCheckpoint(checkpoint: Prefix) -> Prefix`
- `RestoreIfDead(prefix: Prefix, checkpoint: Prefix) -> Prefix`
- `HasBudget(stepsLeft: int, needed: int) -> bool`
- `MinStepsToComplete(prefix: Prefix) -> int`

## Exported Public Surface (`__all__`)

- `MODULE_NAME`
- `Token`, `Prefix`, `Id`, `Logit`
- `DafnySpec`, `dafny_spec`
- `Contains`, `PrefixContains`, `DelimitedAnswerValidForParser`
- `LeftDelimiter`, `RightDelimiter`, `SpacedLeftDelimiter`, `SpacedRightDelimiter`
- `LM`, `Parser`, `Delimiter`, `CSDHelpers`
