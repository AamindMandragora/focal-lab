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
    # Strategy: run a single budget-bounded decoding loop with explicit phases.
    # First, explicitly open the Spider SQL answer span with << using
    # AppendLeftDelimiter inside the loop. Then decode the SQL body primarily with
    # AppendConstrainedStep so the parser can guide generation and allow natural >>
    # closure when available. Maintain a bounded checkpoint at the last known good
    # in-span prefix and use RestoreIfDead / RestoreCheckpoint for lightweight local
    # recovery if a constrained step reaches a dead suffix. If the grammar suffix is
    # complete but >> has not yet been emitted, close the span with
    # AppendRightDelimiter inside the same loop. Stop immediately once >> is the
    # last token, and never emit delimiters outside the decoding loop.
    # CSD_RATIONALE_END
    checkpoint = helpers.Checkpoint(generated)
    attempts = 0
    max_attempts = 2

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        if helpers.EndsWithRightDelimiter(generated):
            break
        elif not helpers.ContainsLeftDelimiter(generated):
            generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
            checkpoint = helpers.Checkpoint(generated)
        elif helpers.IsDead(generated):
            if attempts < max_attempts:
                generated = helpers.RestoreCheckpoint(checkpoint)
                attempts = attempts + 1
                if helpers.EndsWithRightDelimiter(generated):
                    break
                elif helpers.IsComplete(generated):
                    generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                elif helpers.CanConstrain(generated):
                    generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                    generated = helpers.RestoreIfDead(generated, checkpoint)
                    if not helpers.IsDead(generated):
                        checkpoint = helpers.Checkpoint(generated)
                else:
                    break
            else:
                break
        elif helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            generated = helpers.RestoreIfDead(generated, checkpoint)
            if helpers.EndsWithRightDelimiter(generated):
                break
            elif helpers.IsDead(generated):
                if attempts < max_attempts:
                    generated = helpers.RestoreCheckpoint(checkpoint)
                    attempts = attempts + 1
                    if helpers.EndsWithRightDelimiter(generated):
                        break
                    elif helpers.IsComplete(generated):
                        generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                    elif helpers.CanConstrain(generated):
                        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                        generated = helpers.RestoreIfDead(generated, checkpoint)
                        if not helpers.IsDead(generated):
                            checkpoint = helpers.Checkpoint(generated)
                    else:
                        break
                else:
                    break
            else:
                checkpoint = helpers.Checkpoint(generated)
        elif helpers.IsComplete(generated):
            generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
