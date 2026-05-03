# VerifiedAgentSynthesis API Reference

This file documents the helper surface shared by:

- `generation/csd/VerifiedAgentSynthesis.py`
- `generation/csd/VerifiedAgentSynthesis.dfy`

The Python file is the default authoring source. The Dafny file is the checked-in fallback/helper artifact used by the Dafny-first path.

## Types and Constants

- `Token = str`
- `Prefix = list[Token]`
- `Id = int`
- `Logit = float`
- `MODULE_NAME = "VerifiedDecoderAgent"`
- `LeftDelimiter = "<<"`
- `RightDelimiter = ">>"`
- `SpacedLeftDelimiter = " <<"`
- `SpacedRightDelimiter = " >>"`

## Metadata

- `DafnySpec`
  - frozen dataclass holding `kind`, `reads`, `modifies`, `requires`, `ensures`, `decreases`, `axiom`, and `extern`
- `dafny_spec(**kwargs)`
  - decorator that attaches a `DafnySpec` to callables

## Top-Level Predicates

- `Contains(s, sub)`
- `PrefixContains(p, t)`
- `DelimitedAnswerValidForParser(parser, prefix)`
  - checks for a closed delimiter span with non-empty parser-valid content

## `LM`

### Invariant

- `__init__()`
- `ValidTokensIdsLogits()`
- `ValidTokensIdsLogitsAlways()`

### Token / Id / Logit Conversion

- `IdToToken(id)`
- `TokenToId(token)`
- `TokenToIdRecursive(token, offset)`
- `IdToLogit(id)`
- `TokenToLogit(token)`
- `TokensToLogits(tokens)`
- `IdsToLogits(ids)`

### Hard Masking

- `MaskToken(token)`
- `MaskTokens(tokens)`
- `MaskTokensExcept(tokens)`
- `IsMasked(token)`
- `HasUnmaskedToken()`

### Soft Shaping

- `BiasToken(token, delta)`
- `BiasTokens(tokens, delta)`
- `ScaleToken(token, factor)`
- `ScaleTokens(tokens, factor)`
- `ClampLogits(low, high)`

### Filtering and Observation

- `TopKFilter(k)`
- `GetTopKTokens(k)`
- `GetMaxLogitToken()`
- `GetMaxUnmaskedLogit()`
- `GetLogitGap()`
- `SnapshotLogits()`
- `RestoreLogits(snapshot)`

### Generation

- `GenerateLogits(input)`
- `ChooseNextToken()`

## `Parser`

### Core

- `IsValidPrefix(prefix)`
- `EmptyPrefixIsValid()`
- `IsCompletePrefix(prefix)`
- `IsDeadPrefix(prefix)`
- `ValidNextToken(prefix, token)`
- `ValidNextTokens(prefix)`
- `ValidContinuationCount(prefix)`
- `ParserDistanceToComplete(prefix)`

### Extended Queries

- `ValidNextTokensInSet(prefix, candidates)`
- `SharesParserState(prefix_a, prefix_b)`
  - compares parser states by valid-next-token membership rather than raw prefix text

## `Delimiter`

- `__init__(left, right)`
- `LastLeftDelimiterIndex(prefix)`
- `FirstRightDelimiterIndex(content)`
- `GetDelimitedContent(prefix)`
- `InsideDelimitedWindow(prefix)`

### Lemmas

- `NoFirstRightDelimiterIndexMeansNoRight(content)`
- `InsideDelimitedWindowNoRight(prefix)`
- `GetDelimitedContentAppend(prefix, next)`
- `AppendLeftEntersWindow(prefix)`
- `FirstRightDelimiterAppendRight(content)`
- `LastLeftDelimiterAppendNonLeft(prefix, tok)`
- `AppendRightExitsWindow(prefix)`

## `CSDHelpers`

### Construction

- `__init__(lm, parser)`

### Proof Lemmas

- `AllValidNextTokensInLM(content)`
- `ValidNextTokensInLMAfterStep(content, next)`

### Suffix Alignment

- `LongestValidSuffix(prefix)`
- `LongestValidSuffixAppend(prefix, next)`
- `LongestValidSuffixIsValid(prefix)`
- `LongestValidSuffixNotDead(prefix)`

### Grammar State Queries

- `CanConstrain(prefix)`
- `IsComplete(prefix)`
- `IsDead(prefix)`
- `ValidContinuationCount(prefix)`
- `ParserDistanceToComplete(prefix)`
- `ValidTokenCount(prefix)`
- `MinStepsToComplete(prefix)`

All of these route through `LongestValidSuffix`.

### Delimiter Predicates

- `IsLeftDelimiterToken(token)`
- `IsRightDelimiterToken(token)`
- `EndsWithLeftDelimiter(prefix)`
- `EndsWithRightDelimiter(prefix)`
- `ContainsLeftDelimiter(prefix)`
- `ContainsRightDelimiter(prefix)`

### Prefix Scanning

- `LastTokenBefore(generated, target) -> (Token, bool)`
- `CountOccurrences(generated, target) -> int`
- `TokensSinceLastDelimiter(generated) -> int`
- `NgramCount(generated, ngram) -> int`
  - requires `|ngram| >= 1`
- `LastNTokens(generated, n) -> Prefix`
  - returns the last `min(n, |generated|)` tokens
- `FindLastIndex(generated, target) -> int`
  - returns `-1` if the token is absent
- `SliceFrom(generated, start) -> Prefix`
  - requires `0 <= start <= |generated|`
- `SliceRange(generated, start, end) -> Prefix`
  - requires `0 <= start <= end <= |generated|`

### Primitive Step Functions

All step functions return `(nextToken, stepsLeft - 1)`.

- `UnconstrainedStep(prompt, generated, stepsLeft)`
- `UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)`
- `UnconstrainedBiasLeftDelimiterStep(prompt, generated, bias, stepsLeft)`
- `UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)`
- `ConstrainedStep(prompt, generated, stepsLeft)`
- `ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)`
- `SoftConstrainedStep(prompt, generated, penalty, stepsLeft)`
- `TopKConstrainedStep(prompt, generated, k, stepsLeft)`
- `ForcedTokenStep(prompt, generated, token, stepsLeft)`
- `CustomPrefixStep(lmInput, grammarPrefix, stepsLeft)`
  - decouples LM context from grammar state
- `CustomPrefixSoftStep(lmInput, grammarPrefix, penalty, stepsLeft)`
- `CustomPrefixTopKStep(lmInput, grammarPrefix, k, stepsLeft)`

### Logit-Shaping Composites

These are intended to run after `GenerateLogits(...)` and before `ChooseNextToken()`.

- `SoftConstrainToGrammar(prefix, penalty)`
- `IntersectWithGrammar(prefix)`
- `BiasForCompletion(prefix, bonus)`
- `MaskAllDelimiters(generated)`
- `MaskRightDelimiters(generated)`
- `MaskLeftDelimiters(generated)`
- `BiasLeftDelimiters(bias)`
- `BiasRightDelimiters(bias)`
- `BiasTokenGroup(tokens, bonus)`
- `PenalizeTokenGroup(tokens, penalty)`
- `IntersectWithGrammarOnPrefix(grammarPrefix)`
- `SoftConstrainToGrammarOnPrefix(grammarPrefix, penalty)`

### Append Wrappers

All append wrappers return `(updatedPrefix, stepsLeft - 1)`.

- `AppendUnconstrainedStep(prompt, prefix, stepsLeft)`
- `AppendUnconstrainedAllowLeftDelimiterStep(prompt, prefix, stepsLeft)`
- `AppendUnconstrainedNudgeLeftDelimiterStep(prompt, prefix, stepsLeft)`
- `AppendConstrainedStep(prompt, prefix, stepsLeft)`
- `AppendConstrainedOrRightDelimiterStep(prompt, prefix, stepsLeft)`
- `AppendSoftConstrainedStep(prompt, prefix, penalty, stepsLeft)`
- `AppendTopKConstrainedStep(prompt, prefix, k, stepsLeft)`
- `AppendForcedToken(prefix, token, stepsLeft)`
- `AppendLeftDelimiter(prefix, stepsLeft)`
- `AppendRightDelimiter(prefix, stepsLeft)`

### Split-Prefix Span-State Helpers

These helpers manage `(generated, insideSpan, currentConstrained)` when a strategy wants to treat the open answer span as its own working prefix.

- `OpenConstrainedSpan(prefix, stepsLeft)`
- `CloseConstrainedSpan(prefix, currentConstrained, stepsLeft)`
- `AppendConstrainedToken(prefix, currentConstrained, token)`
- `AdaptiveConstrainedStep(prompt, stablePrefix, currentConstrained, validTokenGroups, bonus, narrowThreshold, eosToken, stepsLeft)`
- `GroupBoostedConstrainedStep(prompt, stablePrefix, currentConstrained, validTokenGroups, bonus, stepsLeft)`
- `PenalizedConstrainedStep(prompt, stablePrefix, currentConstrained, penaltyTokens, penalty, stepsLeft)`

### Speculation and Recovery

- `SpeculativeConstrain(prompt, generated, currentConstrained, numTokens, stepsLeft)`
  - returns `(candidateTokens, updatedConstrained, remainingSteps)` without committing to `generated`
- `ScoreCandidate(prompt, generated, candidateTokens) -> Logit`
  - scores a candidate sequence token-by-token using the LM’s current model state
- `Checkpoint(prefix)`
- `RestoreCheckpoint(checkpoint)`
- `RestoreIfDead(prefix, checkpoint)`

### Budget Helpers

- `HasBudget(stepsLeft, needed)`

## `CheckpointStack`

- `__init__()`
- `Push(prefix)`
- `Pop() -> Prefix`
- `Peek() -> Prefix`
- `Depth() -> int`
- `IsEmpty() -> bool`

## `RepetitionTracker`

- `__init__(ngramSize)`
  - requires `ngramSize >= 1`
- `RecordToken(token)`
- `GetCount(ngram) -> int`
  - requires `|ngram| == ngramSize`
- `GetRepetitionPenalty(token) -> Logit`
  - penalty is proportional to the frequency of the matching trailing n-gram
- `ApplyRepetitionPenalties(lm)`

## `__all__`

```python
[
    "MODULE_NAME",
    "Token",
    "Prefix",
    "Id",
    "Logit",
    "DafnySpec",
    "dafny_spec",
    "Contains",
    "PrefixContains",
    "DelimitedAnswerValidForParser",
    "LeftDelimiter",
    "RightDelimiter",
    "SpacedLeftDelimiter",
    "SpacedRightDelimiter",
    "LM",
    "Parser",
    "Delimiter",
    "CSDHelpers",
    "CheckpointStack",
    "RepetitionTracker",
]
```
