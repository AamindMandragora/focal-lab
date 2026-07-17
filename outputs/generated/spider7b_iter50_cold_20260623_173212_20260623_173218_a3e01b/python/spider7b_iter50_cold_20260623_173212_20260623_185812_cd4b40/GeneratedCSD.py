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
        (d_0_helpers_).AppendTaskGuidance(lm, ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating a SQL query. Output SQL: followed by a complete, valid SQL query. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Always include SELECT, FROM, and any required JOIN, WHERE, GROUP BY, ORDER BY clauses. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only table and column names from the schema. Complete the entire query."))))
        d_1_steps_: int
        d_1_steps_ = 0
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_2_og_: _dafny.Seq
            d_3_oi_: bool
            d_4_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_2_og_ = out0_
            d_3_oi_ = out1_
            d_4_oc_ = out2_
            generated = d_2_og_
            insideConstrainedOut = d_3_oi_
            currentConstrainedOut = d_4_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_5_closeReserve_: int
        d_5_closeReserve_ = 99
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_6_groundingBudget_: int = int(0)
            d_7_remaining_: int
            d_7_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_7_remaining_) > (d_5_closeReserve_):
                d_6_groundingBudget_ = (d_7_remaining_) - (d_5_closeReserve_)
            elif True:
                d_6_groundingBudget_ = 0
            if (d_6_groundingBudget_) > (0):
                d_8_stableLen_: int
                d_8_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                d_9_constrainedPrompt_: _dafny.Seq
                d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_8_stableLen_:]))
                d_10_maxStepsPerUnit_: int
                d_10_maxStepsPerUnit_ = 25
                d_11_maxRetries_: int
                d_11_maxRetries_ = 3
                d_12_maxRollbackBudget_: int
                d_12_maxRollbackBudget_ = 10
                if (d_10_maxStepsPerUnit_) > (d_6_groundingBudget_):
                    d_10_maxStepsPerUnit_ = d_6_groundingBudget_
                d_13_resultConstrained_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken, d_10_maxStepsPerUnit_, d_11_maxRetries_, d_12_maxRollbackBudget_)
                d_13_resultConstrained_ = out3_
                generated = (_dafny.SeqWithoutIsStrInference((generated)[:d_8_stableLen_:])) + (d_13_resultConstrained_)
                currentConstrainedOut = d_13_resultConstrained_
                d_1_steps_ = (d_1_steps_) + (d_6_groundingBudget_)
                if (d_1_steps_) > (maxSteps):
                    d_1_steps_ = maxSteps
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_14_closeBudget_: int
            d_14_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_15_cg_: _dafny.Seq
            d_16_ci_: bool
            d_17_cc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
            d_15_cg_ = out4_
            d_16_ci_ = out5_
            d_17_cc_ = out6_
            generated = d_15_cg_
            insideConstrainedOut = d_16_ci_
            currentConstrainedOut = d_17_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

