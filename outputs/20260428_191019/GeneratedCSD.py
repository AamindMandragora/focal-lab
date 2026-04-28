from __future__ import annotations

from VerifiedAgentSynthesis import (
    CSDHelpers,
    LM,
    LeftDelimiter,
    RightDelimiter,
    SpacedLeftDelimiter,
    SpacedRightDelimiter,
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
    # Use natural delimiter-aware reasoning before spans, allow multiple verified arithmetic islands,
    # and adapt closing/extension using phase, span count, completion, continuation count, and budget.
    # CSD_RATIONALE_END
    phase = 0
    verified_spans = 0
    reasoning_tokens = 0
    close_ready = 0
    post_span_tokens = 0
    final_span_goal = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 4:
        if phase == 0:
            next_token = eosToken
            new_steps = stepsLeft
            if verified_spans == 0 and stepsLeft > 8:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            elif stepsLeft > 5:
                next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            else:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == LeftDelimiter or next_token == " <<":
                phase = 1
                close_ready = 0
                final_span_goal = 0
            else:
                reasoning_tokens = reasoning_tokens + 1
                if next_token == "." or next_token == ":" or next_token == ";" or next_token == "\n":
                    close_ready = 1
                elif next_token == "therefore" or next_token == "Therefore" or next_token == "total" or next_token == "Total" or next_token == "answer" or next_token == "Answer":
                    close_ready = 1
                if verified_spans == 0 and close_ready > 0 and stepsLeft <= 12:
                    phase = 0
                elif verified_spans > 0 and post_span_tokens > 0 and close_ready > 0:
                    phase = 0
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            suffix = helpers.LongestValidSuffix(generated)
            if verified_spans == 0 and helpers.CanExtendConstrained(generated) and parser.ValidContinuationCount(suffix) > 0 and stepsLeft > 4 and close_ready == 0:
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
            elif helpers.CanExtendConstrained(generated) and parser.ValidContinuationCount(suffix) > 1 and stepsLeft > helpers.MinStepsToComplete(generated) + 2 and final_span_goal == 0:
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
            else:
                next_token = eosToken
                new_steps = stepsLeft
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == RightDelimiter or next_token == " >>":
                    verified_spans = verified_spans + 1
                    post_span_tokens = 0
                    close_ready = 0
                    if verified_spans >= 2 or stepsLeft <= 2:
                        phase = 3
                    else:
                        phase = 2
                else:
                    phase = 1
        elif phase == 1 and helpers.CanConstrain(generated) and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            if verified_spans == 0 and stepsLeft <= helpers.MinStepsToComplete(generated) + 3:
                generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)
            else:
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        elif phase == 1 and helpers.CanExtendConstrained(generated):
            generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
        elif phase == 2:
            next_token = eosToken
            new_steps = stepsLeft
            if verified_spans == 1 and stepsLeft > 8 and close_ready == 0:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            elif verified_spans == 1 and stepsLeft > 5:
                next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            else:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == LeftDelimiter or next_token == " <<":
                phase = 1
                final_span_goal = 1
                close_ready = 0
            else:
                post_span_tokens = post_span_tokens + 1
                if next_token == "." or next_token == ":" or next_token == ";" or next_token == "\n":
                    close_ready = 1
                elif next_token == "therefore" or next_token == "Therefore" or next_token == "total" or next_token == "Total" or next_token == "answer" or next_token == "Answer":
                    close_ready = 1
                if stepsLeft <= 6:
                    phase = 2
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
