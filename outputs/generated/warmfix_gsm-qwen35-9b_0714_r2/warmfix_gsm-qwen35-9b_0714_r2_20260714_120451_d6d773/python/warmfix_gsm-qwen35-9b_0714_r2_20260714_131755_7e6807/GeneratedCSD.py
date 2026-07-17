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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Think step by step briefly. At the end, write the final symbolic answer expression inside << >> delimiters using only variables and operators (+, -, *, /, //, %, int()). Keep << >> content minimal and exact."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_prefixBudget_: int
        if (maxSteps) > (100):
            d_2_prefixBudget_ = (maxSteps) - (80)
        elif True:
            if (maxSteps) > (20):
                d_2_prefixBudget_ = (maxSteps) - (10)
            elif True:
                d_2_prefixBudget_ = _dafny.euclidian_division(maxSteps, 2)
        d_3_gOut_: _dafny.Seq
        d_4_iOut_: bool
        d_5_cOut_: _dafny.Seq
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out0_, out1_, out2_ = (d_0_helpers_).GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, maxSteps, d_2_prefixBudget_, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
        d_3_gOut_ = out0_
        d_4_iOut_ = out1_
        d_5_cOut_ = out2_
        generated = d_3_gOut_
        insideConstrainedOut = d_4_iOut_
        currentConstrainedOut = d_5_cOut_
        cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

