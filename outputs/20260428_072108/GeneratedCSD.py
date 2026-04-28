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
    # Generate a little free-form text, let the LM naturally decide when to open a delimited span,
    # then keep the span grammar-controlled and let the LM choose the right delimiter only after
    # completion, using budget and continuation-based closing policy.
    # CSD_RATIONALE_END
    phase = 0
    reason_steps = 0
    answer_steps = 0
    close_after = 2
    opened_span = 0
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
            next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            reason_steps = reason_steps + 1
            if next_token == "<<" or next_token == " <<":
                phase = 1
                opened_span = 1
            elif reason_steps >= 4:
                phase = 1
        elif phase == 1 and opened_span == 0:
            next_token = eosToken
            new_steps = stepsLeft
            next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            reason_steps = reason_steps + 1
            if next_token == "<<" or next_token == " <<":
                opened_span = 1
            if opened_span == 1:
                phase = 2
            elif reason_steps >= 8:
                phase = 3
        elif phase == 1 and opened_span == 1:
            if helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                answer_steps = answer_steps + 1
                phase = 2
            else:
                suffix = helpers.LongestValidSuffix(generated)
                if parser.IsCompletePrefix(suffix):
                    next_token = eosToken
                    new_steps = stepsLeft
                    next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                    generated = generated + [next_token]
                    stepsLeft = new_steps
                    if next_token == ">>":
                        phase = 3
                    else:
                        answer_steps = answer_steps + 1
                        phase = 2
                else:
                    break
        elif phase == 2 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            answer_steps = answer_steps + 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            suffix = helpers.LongestValidSuffix(generated)
            if (answer_steps >= close_after and parser.ValidContinuationCount(suffix) <= 1) or stepsLeft <= 2:
                next_token = eosToken
                new_steps = stepsLeft
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == ">>":
                    phase = 3
                else:
                    answer_steps = answer_steps + 1
            elif helpers.CanExtendConstrained(generated):
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                answer_steps = answer_steps + 1
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
