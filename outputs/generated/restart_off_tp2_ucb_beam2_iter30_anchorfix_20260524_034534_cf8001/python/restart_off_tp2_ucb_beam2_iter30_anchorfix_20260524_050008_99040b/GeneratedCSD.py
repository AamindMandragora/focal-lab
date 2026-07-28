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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this gsm_symbolic math word problem. Variables appear with curly braces like {n}, {x}; in math expressions use the BARE name (write n, not {n}).\n\nCRITICAL: End your response with EXACTLY ONE << expression >> at the very end, then stop. Do NOT emit any intermediate << >> spans; intermediate spans cause extraction to fail. Show reasoning in plain prose, then write the single final boxed expression.\n\nFormulation rules:\n(1) Mirror the problem's described ORDER; do not algebraically rewrite. If the problem says 'it takes t minutes per d miles, total y miles', write y//d*t (chunks-first, then rate). Do NOT collapse to (y*t)/d.\n(2) For whole-count discrete answers use // (floor). NEVER add ceiling-division glue like '+ int(x % y > 0)'.\n(3) Apply int(...) ONLY around an individual product that involves a fractional variable (e.g., if w3 means 'three-quarters of a unit', write int(n3*w3) — and keep other terms BARE, like n - (n1*w1) - (n2*w2) - int(n3*w3), not as one big aggregated sum).\n(4) Use ONLY variable names that literally appear in the problem; never invent.\n(5) For percentage answers: int(ratio*100). For unit conversions: 1 ft=12 in, 1 hr=60 min, 1 yr=12 mo, 1 day=24 hr, 1 lb=16 oz.\n\nNeutral example: 'A box holds c cookies. n full boxes plus k loose cookies.' -> Answer expression: <<n*c + k>>.\n\nNow solve, briefly explain, then end with one final <<...>>:")))
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

