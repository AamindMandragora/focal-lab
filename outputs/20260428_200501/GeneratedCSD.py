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
    # Delay opening the verified span until there is clearer natural reasoning evidence: several
    # unconstrained tokens plus punctuation/summary cues, or genuine budget pressure. Then keep the
    # span grammar-controlled and let the LM naturally decide when to close via
    # ConstrainedOrRightDelimiterStep. Prefer a single final span over multiple scratch spans to avoid
    # grading earlier intermediate expressions as final answers.
    # CSD_RATIONALE_END
    phase = 0
    closed_spans = 0
    reason_tokens = 0
    milestones = 0
    summary_cues = 0
    final_ready = 0
    next_token = eosToken
    new_steps = stepsLeft
    suffix = []
    distance = 0
    continuations = 0
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3 and closed_spans < 2:
        if phase == 0:
            if final_ready > 0 or not helpers.HasBudget(stepsLeft, 8):
                next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            elif milestones >= 2 and reason_tokens >= 6:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            elif summary_cues >= 1 and reason_tokens >= 5:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_tokens = reason_tokens + 1
                if len(generated) > 0:
                    next_token = generated[len(generated) - 1]
                    if next_token == "." or next_token == "," or next_token == ":" or next_token == ";" or next_token == "\n":
                        milestones = milestones + 1
                    if next_token == "therefore" or next_token == "Thus" or next_token == "thus" or next_token == "so" or next_token == "total" or next_token == "answer" or next_token == "=":
                        summary_cues = summary_cues + 1
                        milestones = milestones + 1
                    if reason_tokens >= 10 and milestones >= 1:
                        final_ready = 1
                    if reason_tokens >= 14:
                        final_ready = 1
                continue
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == LeftDelimiter or next_token == " <<":
                phase = 1
            else:
                reason_tokens = reason_tokens + 1
                if next_token == "." or next_token == "," or next_token == ":" or next_token == ";" or next_token == "\n":
                    milestones = milestones + 1
                if next_token == "therefore" or next_token == "Thus" or next_token == "thus" or next_token == "so" or next_token == "total" or next_token == "answer" or next_token == "=":
                    summary_cues = summary_cues + 1
                    milestones = milestones + 1
                if reason_tokens >= 10 and milestones >= 1:
                    final_ready = 1
                if reason_tokens >= 14:
                    final_ready = 1
        elif phase == 1 and helpers.CanConstrain(generated):
            suffix = helpers.LongestValidSuffix(generated)
            distance = parser.ParserDistanceToComplete(suffix)
            if distance <= 1:
                generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)
            else:
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            suffix = helpers.LongestValidSuffix(generated)
            continuations = parser.ValidContinuationCount(suffix)
            distance = parser.ParserDistanceToComplete(suffix)
            next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == RightDelimiter or next_token == " >>":
                closed_spans = closed_spans + 1
                phase = 3
            else:
                if continuations <= 1 or not helpers.HasBudget(stepsLeft, 3):
                    final_ready = 1
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
