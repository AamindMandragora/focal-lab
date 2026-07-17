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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating a SQL query. Output ONLY the SQL query, starting with SELECT (or WITH, INSERT, etc.). Do not truncate - always complete the full query with proper WHERE/JOIN/GROUP BY clauses as needed.")))
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
        d_5_closeReserve_ = 8
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_6_stableLen_: int
            d_6_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
            d_7_constrainedPrompt_: _dafny.Seq
            d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_6_stableLen_:]))
            d_8_rolloutBudget_: int
            d_8_rolloutBudget_ = (maxSteps) - (d_1_steps_)
            if (d_8_rolloutBudget_) > (d_5_closeReserve_):
                d_8_rolloutBudget_ = (d_8_rolloutBudget_) - (d_5_closeReserve_)
            elif True:
                d_8_rolloutBudget_ = 0
            if (d_8_rolloutBudget_) > (0):
                d_9_penaltyTokens_: _dafny.Seq
                d_9_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken])
                d_10_penaltyAmount_: _dafny.BigRational
                d_10_penaltyAmount_ = _dafny.BigRational('6e0')
                d_11_rolloutGen_: _dafny.Seq
                d_12_rolloutSteps_: int
                d_13_rolloutEos_: bool
                out3_: _dafny.Seq
                out4_: int
                out5_: bool
                out3_, out4_, out5_ = (d_0_helpers_).RolloutConstrainedWithPenalties(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, d_8_rolloutBudget_, d_9_penaltyTokens_, d_10_penaltyAmount_, eosToken)
                d_11_rolloutGen_ = out3_
                d_12_rolloutSteps_ = out4_
                d_13_rolloutEos_ = out5_
                generated = (_dafny.SeqWithoutIsStrInference((generated)[:d_6_stableLen_:])) + (d_11_rolloutGen_)
                currentConstrainedOut = d_11_rolloutGen_
                d_1_steps_ = (d_1_steps_) + (d_12_rolloutSteps_)
                if (d_1_steps_) > (maxSteps):
                    d_1_steps_ = maxSteps
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_14_closeBudget_: int
            d_14_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_15_cg_: _dafny.Seq
            d_16_ci_: bool
            d_17_cc_: _dafny.Seq
            out6_: _dafny.Seq
            out7_: bool
            out8_: _dafny.Seq
            out6_, out7_, out8_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
            d_15_cg_ = out6_
            d_16_ci_ = out7_
            d_17_cc_ = out8_
            generated = d_15_cg_
            insideConstrainedOut = d_16_ci_
            currentConstrainedOut = d_17_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

