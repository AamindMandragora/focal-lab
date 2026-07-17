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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a concise SQL query. Use simple SELECT FROM WHERE patterns. Avoid aliases (AS), avoid complex subqueries when simple joins or aggregates suffice. Use exact column and table names from the schema. Prefer: SELECT col FROM table WHERE condition. Use LIMIT 1 for single-value queries.")))
        if insideConstrainedOut:
            d_1_closeBudget_: int
            d_1_closeBudget_ = maxSteps
            if (d_1_closeBudget_) > (0):
                d_2_cg_: _dafny.Seq
                d_3_ci_: bool
                d_4_cc_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_1_closeBudget_)
                d_2_cg_ = out0_
                d_3_ci_ = out1_
                d_4_cc_ = out2_
                generated = d_2_cg_
                insideConstrainedOut = d_3_ci_
                currentConstrainedOut = d_4_cc_
        if (not(insideConstrainedOut)) and ((maxSteps) > (0)):
            d_5_remainingBudget_: int
            d_5_remainingBudget_ = maxSteps
            d_6_constrainedGenerated_: _dafny.Seq
            d_7_terminatedByEos_: bool
            out3_: _dafny.Seq
            out4_: bool
            out3_, out4_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, d_5_remainingBudget_, eosToken)
            d_6_constrainedGenerated_ = out3_
            d_7_terminatedByEos_ = out4_
            generated = (generatedPrefix) + (d_6_constrainedGenerated_)
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

