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
        d_1_stoppedOnOpenSpan_: bool = False
        d_2_stoppedOnEos_: bool = False
        d_3_stepsUsed_: int = int(0)
        out0_: _dafny.Seq
        out1_: bool
        out2_: bool
        out3_: int
        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, maxSteps, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
        generated = out0_
        d_1_stoppedOnOpenSpan_ = out1_
        d_2_stoppedOnEos_ = out2_
        d_3_stepsUsed_ = out3_
        cost = d_3_stepsUsed_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

