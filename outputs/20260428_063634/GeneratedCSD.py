from __future__ import annotations

from VerifiedAgentSynthesis import (
    CSDHelpers,
    LM,
    LeftDelimiter,
    RightDelimiter,
    Parser,
    Prefix,
    Token,
    dafny_spec,
)


DAFNY_INCLUDE = "VerifiedAgentSynthesis.dfy"
MODULE_NAME = "GeneratedCSD"
DAFNY_OPEN_IMPORT = "VerifiedDecoderAgent"


@dafny_spec(
    kind="method",
    modifies=("lm.Logits",),
    requires=(
        "lm.ValidTokensIdsLogits()",
        "parser.IsValidPrefix([])",
        "!parser.IsCompletePrefix([])",
        "forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens",
        "maxSteps >= 2",
        "LeftDelimiter in lm.Tokens",
        "RightDelimiter in lm.Tokens",
    ),
    ensures=(
        "lm.ValidTokensIdsLogits()",
        "|generated| <= maxSteps",
        "remainingSteps >= 0 && remainingSteps <= maxSteps",
    ),
)
def MyCSDStrategy(
    lm: LM,
    parser: Parser,
    prompt: Prefix,
    maxSteps: int,
    eosToken: Token,
) -> tuple[Prefix, int]:
    helpers = CSDHelpers(lm, parser)
    lm.ValidTokensIdsLogitsAlways()
    generated = []
    stepsLeft = maxSteps
    # CSD_RATIONALE_BEGIN
    # Use adaptive free-form reasoning, then open one verified span inside the loop, grow it with hard constrained steps, optionally extend after first completion, and close only when completion plus budget/continuation signals say the span is ready.
    # CSD_RATIONALE_END
    phase = 0
    reasoning_seen = 0
    span_tokens = 0
    complete_seen = 0
    closed_spans = 0
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3:
        if phase == 0:
            if reasoning_seen == 0:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reasoning_seen = reasoning_seen + 1
            elif closed_spans == 0 and helpers.HasBudget(stepsLeft, helpers.MinStepsToComplete(generated) + 2):
                generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
                phase = 1
                span_tokens = 0
                complete_seen = 0
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reasoning_seen = reasoning_seen + 1
        elif phase == 1 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            span_tokens = span_tokens + 1
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            if helpers.CanExtendConstrained(generated):
                suffix = helpers.LongestValidSuffix(generated)
                continuation_count = parser.ValidContinuationCount(suffix)
                distance_left = parser.ParserDistanceToComplete(suffix)
                if complete_seen == 0 and continuation_count > 0 and stepsLeft > distance_left + 1:
                    generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                    span_tokens = span_tokens + 1
                    complete_seen = complete_seen + 1
                elif continuation_count > 1 and span_tokens < reasoning_seen and stepsLeft > 1:
                    generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                    span_tokens = span_tokens + 1
                    complete_seen = complete_seen + 1
                else:
                    generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                    phase = 2
                    closed_spans = closed_spans + 1
            else:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 2
                closed_spans = closed_spans + 1
        elif phase == 2:
            if stepsLeft > 0 and eosToken in parser.ValidNextTokens(helpers.LongestValidSuffix(generated)):
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reasoning_seen = reasoning_seen + 1
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reasoning_seen = reasoning_seen + 1
                phase = 3
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
