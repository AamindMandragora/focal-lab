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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write SQL without column aliases. Do not use AS keyword. Do not use NOT IN subqueries. Use INTERSECT for set intersection. Use JOIN for multi-table queries. Write the most direct, simple SQL for the question.")))
        if (insideConstrainedOut) and ((maxSteps) > (0)):
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
            cost = d_1_closeBudget_
        if (not(insideConstrainedOut)) and ((maxSteps) > (cost)):
            d_5_penaltyTokens_: _dafny.Seq
            d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " as"))])
            d_6_remainingBudget_: int
            d_6_remainingBudget_ = (maxSteps) - (cost)
            d_7_rolloutGen_: _dafny.Seq
            d_8_rolloutSteps_: int
            d_9_rolloutEos_: bool
            out3_: _dafny.Seq
            out4_: int
            out5_: bool
            out3_, out4_, out5_ = (d_0_helpers_).RolloutConstrainedWithPenalties(lm, parser, prompt, _dafny.SeqWithoutIsStrInference([]), d_6_remainingBudget_, d_5_penaltyTokens_, _dafny.BigRational('6e0'), eosToken)
            d_7_rolloutGen_ = out3_
            d_8_rolloutSteps_ = out4_
            d_9_rolloutEos_ = out5_
            generated = (generatedPrefix) + (d_7_rolloutGen_)
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            cost = (cost) + (d_8_rolloutSteps_)
        if ((cost) == (0)) and ((maxSteps) > (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

