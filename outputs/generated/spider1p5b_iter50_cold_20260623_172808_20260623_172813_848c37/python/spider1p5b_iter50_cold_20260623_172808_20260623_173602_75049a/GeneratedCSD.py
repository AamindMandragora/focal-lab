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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SQL query. Use SELECT with the correct table and column names from the schema. Answer the question directly using SQL syntax only."))
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
                cost = d_2_closeBudget_
        elif True:
            if (maxSteps) > (0):
                d_6_constrainedGenerated_: _dafny.Seq
                d_7_terminatedByEos_: bool
                out3_: _dafny.Seq
                out4_: bool
                out3_, out4_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken)
                d_6_constrainedGenerated_ = out3_
                d_7_terminatedByEos_ = out4_
                generated = (generatedPrefix) + (d_6_constrainedGenerated_)
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

