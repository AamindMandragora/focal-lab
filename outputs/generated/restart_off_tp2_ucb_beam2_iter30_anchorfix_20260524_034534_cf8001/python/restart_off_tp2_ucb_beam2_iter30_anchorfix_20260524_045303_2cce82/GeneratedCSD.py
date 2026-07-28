import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_
import VerifiedDecoderAgent as VerifiedDecoderAgent

# Module: GeneratedCSD

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are solving a symbolic math word problem. Variables appear in the question with curly braces like {n}, {x}, {name}; inside any math expression use the BARE name without braces (write n, not {n}). After a brief explanation, write EXACTLY ONE << final_formula >> at the end of your response and stop. Do NOT emit intermediate << >> spans for sub-steps; the final formula is the only << >>. Rules: (1) Use ONLY variable names that appear in this specific problem; never invent new variables. (2) For answers that must be a whole count of discrete items, use // (Python integer division), not /. (3) For percentage answers, multiply the ratio by 100 and wrap with int(...). (4) Common unit conversions: 1 foot = 12 inches, 1 hour = 60 minutes, 1 year = 12 months, 1 day = 24 hours. (5) Do not add ceiling-division glue such as `+ int(x % y > 0)`; just write the direct formula the question asks for. (6) Re-read the question to identify which named quantity is the final answer. Solve concisely now:")))
        while (cost) < (maxSteps):
            d_1_remaining_: int
            d_1_remaining_ = (maxSteps) - (cost)
            d_2_newGenerated_: _dafny.Seq
            d_3_stoppedOnOpenSpan_: bool
            d_4_stoppedOnEos_: bool
            d_5_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_1_remaining_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_2_newGenerated_ = out0_
            d_3_stoppedOnOpenSpan_ = out1_
            d_4_stoppedOnEos_ = out2_
            d_5_stepsUsed_ = out3_
            generated = d_2_newGenerated_
            cost = (cost) + (d_5_stepsUsed_)
            if d_4_stoppedOnEos_:
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if (d_5_stepsUsed_) == (0):
                return generated, insideConstrainedOut, currentConstrainedOut, cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

