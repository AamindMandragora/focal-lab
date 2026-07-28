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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For every arithmetic operation, write it inline as <<expr=value>> using the literal delimiters << and >>, e.g. <<3+4=7>>. Always close each << with a matching >>. After reasoning, write '#### ' followed by the final numeric answer on its own line.")))
        d_1_minReasoning_: int
        d_1_minReasoning_ = 1
        d_2_craneGen_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = (d_0_helpers_).CraneGeneration(lm, parser, (prompt) + (generatedPrefix), maxSteps, d_1_minReasoning_, eosToken)
        d_2_craneGen_ = out0_
        generated = (generatedPrefix) + (d_2_craneGen_)
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

