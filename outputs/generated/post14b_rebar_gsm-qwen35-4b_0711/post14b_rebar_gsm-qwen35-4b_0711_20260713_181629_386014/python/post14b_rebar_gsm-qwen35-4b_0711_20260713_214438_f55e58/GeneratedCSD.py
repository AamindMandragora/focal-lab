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
        if (maxSteps) == (0):
            pass
        elif True:
            d_1_guidance_: _dafny.Seq
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step using plain text reasoning. At the very end, write your final arithmetic expression inside << >> exactly once. The expression inside << >> must be a COMPLETE arithmetic expression using the variable names from the problem (e.g., <<n * (m + k)>>). Use only: variable names, numbers, +, -, *, /, //, %, (, ), int(). Do NOT open << >> anywhere except for the single final answer. Make sure to include ALL variables needed for the complete answer inside the single << >> span."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_prefixBudget_: int
            d_2_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (17), 20)
            if (d_2_prefixBudget_) >= (maxSteps):
                d_2_prefixBudget_ = (maxSteps) - (1)
            d_3_gFinal_: _dafny.Seq
            d_4_icFinal_: bool
            d_5_ccFinal_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, maxSteps, d_2_prefixBudget_, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
            d_3_gFinal_ = out0_
            d_4_icFinal_ = out1_
            d_5_ccFinal_ = out2_
            generated = d_3_gFinal_
            insideConstrainedOut = d_4_icFinal_
            currentConstrainedOut = d_5_ccFinal_
            cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

