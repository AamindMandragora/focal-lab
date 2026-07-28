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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a SQL query to answer the question. Use only exact table and column names from the schema. Use the simplest correct SQL. Prefer direct joins using the correct join columns. Carefully identify which direction the join goes (source vs destination)."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if insideConstrainedOut:
            if (maxSteps) > (0):
                d_2_closeBudget_: int
                d_2_closeBudget_ = maxSteps
                d_3_cg_: _dafny.Seq
                d_4_ci_: bool
                d_5_cc_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_2_closeBudget_)
                d_3_cg_ = out0_
                d_4_ci_ = out1_
                d_5_cc_ = out2_
                generated = d_3_cg_
                insideConstrainedOut = d_4_ci_
                currentConstrainedOut = d_5_cc_
                cost = maxSteps
        elif True:
            if (maxSteps) > (0):
                d_6_maxStepsPerUnit_: int
                if (maxSteps) >= (4):
                    d_6_maxStepsPerUnit_ = _dafny.euclidian_division(maxSteps, 4)
                elif True:
                    d_6_maxStepsPerUnit_ = 1
                d_7_resultConstrained_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, prompt, _dafny.SeqWithoutIsStrInference([]), eosToken, d_6_maxStepsPerUnit_, 3, 10)
                d_7_resultConstrained_ = out3_
                generated = (generatedPrefix) + (d_7_resultConstrained_)
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

