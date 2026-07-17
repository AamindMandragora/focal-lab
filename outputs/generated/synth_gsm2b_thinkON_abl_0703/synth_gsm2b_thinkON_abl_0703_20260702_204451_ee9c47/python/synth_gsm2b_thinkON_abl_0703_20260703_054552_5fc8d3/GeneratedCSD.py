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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step using the variable names from the problem. After completing your reasoning, write the final algebraic expression inside << >> delimiters using plain variable names without curly braces and without LaTeX. Use only these operators: +, -, *, /, //, %, int(), and parentheses. Examples: <<n * (mult + 1)>>, <<total - int(total * frac) - daily * period * 7>>, <<n - n1*w1 - n2*w2 - n3*w3>>, <<int(n * p1/100 * p2/100 * frac)>>. Use the exact variable names as they appear in the problem template.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_prefixBudget_: int
            d_1_prefixBudget_ = _dafny.euclidian_division((maxSteps) * (4), 5)
            if (d_1_prefixBudget_) >= (maxSteps):
                d_1_prefixBudget_ = (maxSteps) - (1)
            d_2_gOut_: _dafny.Seq
            d_3_iOut_: bool
            d_4_cOut_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, maxSteps, d_1_prefixBudget_, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
            d_2_gOut_ = out0_
            d_3_iOut_ = out1_
            d_4_cOut_ = out2_
            generated = d_2_gOut_
            insideConstrainedOut = d_3_iOut_
            currentConstrainedOut = d_4_cOut_
            cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

