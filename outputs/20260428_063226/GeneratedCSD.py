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
    # Generate a short free-form lead-in, then open one delimited constrained span, grow it with
    # hard constrained steps and optional extend steps, and close it only in a completion-guarded
    # branch after at least one constrained token has been emitted.
    # CSD_RATIONALE_END
    phase = 0
    reason_steps = 0
    answer_tokens = 0
    close_score = 0
    reason_limit = 2

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 4:
        if phase == 0 and reason_steps < reason_limit and stepsLeft > 0:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reason_steps = reason_steps + 1
            if reason_steps >= reason_limit:
                phase = 1
        elif phase == 0 and stepsLeft > 0:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            phase = 1
        elif phase == 1 and stepsLeft > 0:
            generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
            phase = 2
        elif phase == 2 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            answer_tokens = answer_tokens + 1
            if answer_tokens >= 2:
                close_score = close_score + 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)) and helpers.CanExtendConstrained(generated) and answer_tokens < 3:
            generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
            answer_tokens = answer_tokens + 1
            close_score = close_score + 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)) and helpers.CanExtendConstrained(generated) and parser.ValidContinuationCount(helpers.LongestValidSuffix(generated)) > 1 and helpers.MinStepsToComplete(generated) + 1 < stepsLeft and close_score < 2:
            generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
            answer_tokens = answer_tokens + 1
            close_score = close_score + 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)) and answer_tokens > 0 and (close_score > 0 or stepsLeft <= 2 or parser.ValidContinuationCount(helpers.LongestValidSuffix(generated)) <= 1):
            generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
            phase = 3
        elif phase == 3 and stepsLeft > 0:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            phase = 4
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
