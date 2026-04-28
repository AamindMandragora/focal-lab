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
    # Use a budget-bounded phase machine: begin with free-form reasoning that naturally allows a left delimiter, switch into a constrained span only after the LM actually emits the delimiter, keep growing the grammar-valid answer under explicit guards, and let the LM choose the right delimiter once completion is available and closing is justified by budget/length signals.
    # CSD_RATIONALE_END
    phase = 0
    reason_steps = 0
    answer_steps = 0
    close_score = 0
    saw_left = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3:
        if phase == 0:
            next_token = eosToken
            new_steps = stepsLeft
            if stepsLeft <= 3 or reason_steps >= 4:
                next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            else:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            reason_steps = reason_steps + 1
            if next_token == "<<" or next_token == " <<":
                phase = 1
                saw_left = 1
            else:
                phase = 0
        elif phase == 1 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            answer_steps = answer_steps + 1
            phase = 1
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            suffix = helpers.LongestValidSuffix(generated)
            conts = parser.ValidContinuationCount(suffix)
            dist = parser.ParserDistanceToComplete(suffix)
            if answer_steps >= 3:
                close_score = close_score + 1
            if conts <= 1:
                close_score = close_score + 1
            if stepsLeft <= 2:
                close_score = close_score + 1
            if close_score >= 2:
                next_token = eosToken
                new_steps = stepsLeft
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == ">>":
                    phase = 2
                else:
                    answer_steps = answer_steps + 1
                    phase = 1
            elif helpers.CanExtendConstrained(generated):
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                answer_steps = answer_steps + 1
                phase = 1
            else:
                next_token = eosToken
                new_steps = stepsLeft
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == ">>":
                    phase = 2
                else:
                    answer_steps = answer_steps + 1
                    phase = 1
        else:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            phase = 2
    remainingSteps = stepsLeft
    return generated, remainingSteps
