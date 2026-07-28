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
        d_1_craneBudget_: int
        d_1_craneBudget_ = maxSteps
        d_2_minReasoning_: int
        if (d_1_craneBudget_) >= (8):
            d_2_minReasoning_ = 8
        elif True:
            d_2_minReasoning_ = 0
        d_3_craneGenerated_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = (d_0_helpers_).CraneGeneration(lm, parser, (prompt) + (generated), d_1_craneBudget_, d_2_minReasoning_, eosToken)
        d_3_craneGenerated_ = out0_
        if (len(d_3_craneGenerated_)) > (d_1_craneBudget_):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference((d_3_craneGenerated_)[:d_1_craneBudget_:]))
        elif True:
            generated = (generated) + (d_3_craneGenerated_)
        cost = d_1_craneBudget_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

