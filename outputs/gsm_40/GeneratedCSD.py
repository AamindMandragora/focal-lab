from __future__ import annotations

from VerifiedAgentSynthesis import (
    CSDHelpers,
    LM,
    LeftDelimiter,
    RightDelimiter,
    SpacedLeftDelimiter,
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
    # Generate a little free-form reasoning, naturally nudge the model to open a delimiter, then keep all tokens inside the span grammar-controlled and let the model naturally choose the right delimiter once the parse is complete and budget-aware closing is appropriate.
    # CSD_RATIONALE_END
    phase = 0
    reason_steps = 0
    answer_steps = 0
    close_after_complete = 0
    opened_span = 0
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3:
        if phase == 0 and reason_steps < 2:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reason_steps = reason_steps + 1
            if reason_steps >= 2:
                phase = 1
        elif phase == 0:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reason_steps = reason_steps + 1
            phase = 1
        elif phase == 1:
            next_token = eosToken
            new_steps = stepsLeft
            if stepsLeft <= 3:
                next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            else:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == "<<" or next_token == " <<":
                opened_span = 1
                phase = 2
            else:
                reason_steps = reason_steps + 1
                if reason_steps >= 4:
                    phase = 1
                else:
                    phase = 0
        elif phase == 2 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            answer_steps = answer_steps + 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            suffix = helpers.LongestValidSuffix(generated)
            continuation_count = parser.ValidContinuationCount(suffix)
            distance = parser.ParserDistanceToComplete(suffix)
            if answer_steps >= 2 and (stepsLeft <= 2 or continuation_count <= 1 or close_after_complete > 0 or distance == 0):
                next_token = eosToken
                new_steps = stepsLeft
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == ">>":
                    phase = 3
                else:
                    answer_steps = answer_steps + 1
                    close_after_complete = close_after_complete + 1
            elif helpers.CanExtendConstrained(generated):
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                answer_steps = answer_steps + 1
                close_after_complete = close_after_complete + 1
            else:
                next_token = eosToken
                new_steps = stepsLeft
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == ">>":
                    phase = 3
                else:
                    answer_steps = answer_steps + 1
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
