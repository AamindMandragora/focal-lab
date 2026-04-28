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
    # Use a phased strategy: first generate ordinary reasoning, then only after moderate reasoning
    # progress or budget pressure nudge a natural left delimiter, then stay grammar-controlled inside
    # the span until a complete answer is rich enough to close with a natural right delimiter.
    # CSD_RATIONALE_END
    phase = 0
    reason_steps = 0
    final_ready = 0
    closed_spans = 0
    span_steps = 0
    next_token = eosToken
    new_steps = stepsLeft
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3 and closed_spans < 2:
        if phase == 0:
            if final_ready == 0 and reason_steps < 10 and helpers.HasBudget(stepsLeft, 8):
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_steps = reason_steps + 1
            elif final_ready == 0:
                next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == LeftDelimiter or next_token == SpacedLeftDelimiter:
                    phase = 1
                    final_ready = 1
                    span_steps = 0
                else:
                    reason_steps = reason_steps + 1
                    if reason_steps >= 10 or not helpers.HasBudget(stepsLeft, 8):
                        final_ready = 1
            else:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == LeftDelimiter or next_token == SpacedLeftDelimiter:
                    phase = 1
                    span_steps = 0
                else:
                    reason_steps = reason_steps + 1
        elif phase == 1 and helpers.CanConstrain(generated):
            suffix = helpers.LongestValidSuffix(generated)
            if span_steps >= 2 and parser.IsCompletePrefix(suffix) and parser.ValidContinuationCount(suffix) <= 1:
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == RightDelimiter or next_token == SpacedRightDelimiter:
                    closed_spans = closed_spans + 1
                    phase = 2
                else:
                    span_steps = span_steps + 1
            elif span_steps >= 4 and parser.IsCompletePrefix(suffix) and not helpers.HasBudget(stepsLeft, 4):
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == RightDelimiter or next_token == SpacedRightDelimiter:
                    closed_spans = closed_spans + 1
                    phase = 2
                else:
                    span_steps = span_steps + 1
            else:
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                span_steps = span_steps + 1
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == RightDelimiter or next_token == SpacedRightDelimiter:
                closed_spans = closed_spans + 1
                phase = 2
            else:
                span_steps = span_steps + 1
        elif phase == 2:
            break
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
