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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. For each arithmetic operation and the final answer, write the expression inside << >> delimiters. Use only numbers, +, -, *, /, (, ) inside the delimiters. The final answer must be inside << >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_prefixBudget_: int
        if (maxSteps) >= (25):
            d_2_prefixBudget_ = (maxSteps) - (20)
        elif True:
            if (maxSteps) >= (5):
                d_2_prefixBudget_ = (maxSteps) - (4)
            elif True:
                d_2_prefixBudget_ = maxSteps
        d_3_boostAmount_: _dafny.BigRational
        d_3_boostAmount_ = _dafny.BigRational('6e0')
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 8
        d_5_resultGenerated_: _dafny.Seq
        d_6_resultInside_: bool
        d_7_resultCurrent_: _dafny.Seq
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out0_, out1_, out2_ = (d_0_helpers_).GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, maxSteps, d_2_prefixBudget_, validTokenGroups, d_3_boostAmount_, d_4_narrowThreshold_, eosToken)
        d_5_resultGenerated_ = out0_
        d_6_resultInside_ = out1_
        d_7_resultCurrent_ = out2_
        generated = d_5_resultGenerated_
        insideConstrainedOut = d_6_resultInside_
        currentConstrainedOut = d_7_resultCurrent_
        cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

