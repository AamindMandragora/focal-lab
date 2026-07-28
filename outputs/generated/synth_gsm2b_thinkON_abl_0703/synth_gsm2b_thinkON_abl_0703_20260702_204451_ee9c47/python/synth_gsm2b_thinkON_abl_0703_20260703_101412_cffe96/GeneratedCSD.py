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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. The variables in {braces} are symbolic placeholders. Reason about them algebraically. At the very end of your solution, write the final answer inside << >> delimiters. The answer inside << >> must be a single number (integer or decimal), for example: <<42>> or <<3.5>>. Do not include variables, expressions, or text inside << >>. Compute the numeric result first, then write it inside << >>.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_prefixBudget_: int
            d_1_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (82), 100)
            if (d_1_prefixBudget_) >= (maxSteps):
                d_1_prefixBudget_ = (maxSteps) - (1)
            d_2_gOut_: _dafny.Seq
            d_3_iOut_: bool
            d_4_cOut_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, maxSteps, d_1_prefixBudget_, validTokenGroups, _dafny.BigRational('4e0'), 8, eosToken)
            d_2_gOut_ = out0_
            d_3_iOut_ = out1_
            d_4_cOut_ = out2_
            generated = d_2_gOut_
            insideConstrainedOut = d_3_iOut_
            currentConstrainedOut = d_4_cOut_
            cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

