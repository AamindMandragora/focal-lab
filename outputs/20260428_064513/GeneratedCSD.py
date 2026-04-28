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
    # Use adaptive interleaving: begin with free-form reasoning, deliberately open one verified span,
    # grow it with hard grammar control, and close only when the parse is complete and either budget
    # pressure or semantic richness signals that closing is appropriate.
    # CSD_RATIONALE_END
    phase = 0
    reasoning_seen = 0
    answer_tokens = 0
    completed_once = 0
    post_span_tokens = 0
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 4:
        if phase == 0:
            if reasoning_seen == 0:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reasoning_seen = reasoning_seen + 1
            elif helpers.HasBudget(stepsLeft, helpers.MinStepsToComplete(generated) + 2):
                generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
                phase = 1
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reasoning_seen = reasoning_seen + 1
        elif phase == 1:
            if helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                answer_tokens = answer_tokens + 1
                phase = 2
            else:
                break
        elif phase == 2 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            answer_tokens = answer_tokens + 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            if helpers.CanExtendConstrained(generated) and answer_tokens < 3 and helpers.HasBudget(stepsLeft, 2):
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                answer_tokens = answer_tokens + 1
                completed_once = 1
            elif helpers.CanExtendConstrained(generated) and parser.ValidContinuationCount(helpers.LongestValidSuffix(generated)) > 1 and helpers.HasBudget(stepsLeft, 2) and completed_once == 0:
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                answer_tokens = answer_tokens + 1
                completed_once = 1
            else:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 3
        elif phase == 3:
            if stepsLeft > 1 and post_span_tokens == 0:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                post_span_tokens = post_span_tokens + 1
            else:
                phase = 4
                break
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
