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
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single SQL query. Use lowercase keywords. Write parentheses with spaces: count ( * ) not count(*). Use simple subqueries with intersect/except/union when needed. Do not use JOIN if a simple WHERE with subquery suffices. Output only the SQL query inside << >>.")))
        if (d_1_steps_) < (maxSteps):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
            d_1_steps_ = (d_1_steps_) + (1)
        d_2_stoppedOnClose_: bool
        d_2_stoppedOnClose_ = False
        if (d_1_steps_) < (maxSteps):
            d_3_budget_: int
            d_3_budget_ = (maxSteps) - (d_1_steps_)
            d_4_gOut_: _dafny.Seq
            d_5_sOpen_: bool
            d_6_sEos_: bool
            d_7_used_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), eosToken)
            d_4_gOut_ = out0_
            d_5_sOpen_ = out1_
            d_6_sEos_ = out2_
            d_7_used_ = out3_
            generated = d_4_gOut_
            d_1_steps_ = (d_1_steps_) + (d_7_used_)
            d_2_stoppedOnClose_ = d_5_sOpen_
        if (not(d_2_stoppedOnClose_)) and ((d_1_steps_) < (maxSteps)):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

