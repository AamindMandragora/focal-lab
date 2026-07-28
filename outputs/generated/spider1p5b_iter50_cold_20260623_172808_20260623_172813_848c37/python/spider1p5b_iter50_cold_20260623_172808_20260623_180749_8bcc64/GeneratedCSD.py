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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write the simplest possible SQL query to answer the question. Use INTERSECT when the question says 'both' two groups. Do not use table aliases like T1/T2. Do not use INNER JOIN unless required. Use exact column and table names from schema. Example: SELECT col FROM table WHERE condition"))
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
                d_6_aliasTokens_: _dafny.Seq
                d_6_aliasTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "INNER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "inner"))])
                (d_0_helpers_).SafePenalizeTokenLogits(lm, d_6_aliasTokens_, _dafny.BigRational('3e0'))
                d_7_constrainedGenerated_: _dafny.Seq
                d_8_terminatedByEos_: bool
                out3_: _dafny.Seq
                out4_: bool
                out3_, out4_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken)
                d_7_constrainedGenerated_ = out3_
                d_8_terminatedByEos_ = out4_
                generated = (generatedPrefix) + (d_7_constrainedGenerated_)
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

