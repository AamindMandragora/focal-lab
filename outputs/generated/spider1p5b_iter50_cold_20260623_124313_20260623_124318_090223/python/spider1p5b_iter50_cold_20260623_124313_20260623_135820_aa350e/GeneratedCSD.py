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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a minimal SQL query. Use exact table and column names from the schema. No aliases (no AS, no t1., t2.). For aggregations: SELECT MAX(col) FROM table. For filtering: SELECT col FROM table WHERE condition. Keep queries simple and direct.")))
        if (maxSteps) == (0):
            pass
        elif insideConstrainedOut:
            d_1_closeBudget_: int
            d_1_closeBudget_ = maxSteps
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
            cost = maxSteps
        elif True:
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out3_: _dafny.Seq
            out4_: bool
            out5_: _dafny.Seq
            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out3_
            d_6_oi_ = out4_
            d_7_oc_ = out5_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_8_closeBudget_: int
            d_8_closeBudget_ = (maxSteps) - (1)
            if (d_8_closeBudget_) > (0):
                d_9_cg_: _dafny.Seq
                d_10_ci_: bool
                d_11_cc_: _dafny.Seq
                out6_: _dafny.Seq
                out7_: bool
                out8_: _dafny.Seq
                out6_, out7_, out8_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_closeBudget_)
                d_9_cg_ = out6_
                d_10_ci_ = out7_
                d_11_cc_ = out8_
                generated = d_9_cg_
                insideConstrainedOut = d_10_ci_
                currentConstrainedOut = d_11_cc_
            cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

