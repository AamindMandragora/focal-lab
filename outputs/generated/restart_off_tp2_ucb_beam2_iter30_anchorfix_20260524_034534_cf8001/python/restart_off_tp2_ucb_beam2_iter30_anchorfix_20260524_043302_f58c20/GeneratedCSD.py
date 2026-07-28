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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are solving a symbolic math word problem. Variable names like {n}, {x}, {name} are placeholders; use the bare variable names (without braces) inside expressions. For each arithmetic step, wrap the expression in << and >>. Rules: use // (Python integer division) when the answer must be a whole count of discrete items; use * 100 for percentages and wrap with int(...) if the answer is an integer percentage; convert units carefully (1 foot = 12 inches, 1 hour = 60 minutes); read the question carefully to identify which quantity is the final answer. Place the final symbolic answer as the LAST << >> in your response. Example: 'If a {x}-foot rope is cut into {n} pieces of equal length in inches, each piece is <<x * 12 // n>> inches.' Now solve concisely:")))
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

