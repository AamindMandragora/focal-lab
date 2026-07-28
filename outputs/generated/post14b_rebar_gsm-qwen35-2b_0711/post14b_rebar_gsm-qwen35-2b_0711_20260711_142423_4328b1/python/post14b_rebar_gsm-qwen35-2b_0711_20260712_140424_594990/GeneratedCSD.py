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
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are solving a math word problem. Read the problem and directly write the answer as a Python arithmetic expression using the variable names from the problem. Format: <<EXPRESSION>>. Use only +, -, *, /, //, %, int(), parentheses. No braces, no $, no **. Think briefly then give the formula. Examples: If each person gets n apples and there are k people: <<n * k>>. If you start with total and spend rate per day for days: <<total - rate * days>>. If fraction of n items: <<int(n * frac)>>. If items cost price each and you have money: <<money - n * price>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_prefixBudget_: int
        if (maxSteps) >= (60):
            d_2_prefixBudget_ = 60
        elif True:
            d_2_prefixBudget_ = maxSteps
        d_3_genOut_: _dafny.Seq
        d_4_insideOut_: bool
        d_5_currentOut_: _dafny.Seq
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out0_, out1_, out2_ = (d_0_helpers_).GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, maxSteps, d_2_prefixBudget_, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
        d_3_genOut_ = out0_
        d_4_insideOut_ = out1_
        d_5_currentOut_ = out2_
        generated = d_3_genOut_
        insideConstrainedOut = d_4_insideOut_
        currentConstrainedOut = d_5_currentOut_
        cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

