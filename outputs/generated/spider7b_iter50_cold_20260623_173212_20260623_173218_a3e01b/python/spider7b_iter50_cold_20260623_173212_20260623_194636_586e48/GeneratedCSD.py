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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a complete, syntactically valid SQL query using only the table and column names from the provided schema. Always include FROM clause and required JOINs. Complete the full query.")))
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
        d_5_closeReserve_ = 120
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
                d_10_maxStepsPerUnit_ = 20
                d_11_maxRetries_: int
                d_11_maxRetries_ = 2
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
        d_14_flatTokens_: _dafny.Seq
        out4_: _dafny.Seq
        out4_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_14_flatTokens_ = out4_
        with _dafny.label("0"):
            while (insideConstrainedOut) and (((d_1_steps_) + (2)) <= (maxSteps)):
                with _dafny.c_label("0"):
                    d_15_cg_: _dafny.Seq
                    d_16_ci_: bool
                    d_17_cc_: _dafny.Seq
                    d_18_closed_: bool
                    out5_: _dafny.Seq
                    out6_: bool
                    out7_: _dafny.Seq
                    out8_: bool
                    out5_, out6_, out7_, out8_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_15_cg_ = out5_
                    d_16_ci_ = out6_
                    d_17_cc_ = out7_
                    d_18_closed_ = out8_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_18_closed_:
                        generated = d_15_cg_
                        insideConstrainedOut = d_16_ci_
                        currentConstrainedOut = d_17_cc_
                        raise _dafny.Break("0")
                    (d_0_helpers_).SafeBoostTokenLogits(lm, d_14_flatTokens_, _dafny.BigRational('6e0'))
                    d_19_stableLen_: int
                    d_19_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                    d_20_constrainedPrompt_: _dafny.Seq
                    d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_19_stableLen_:]))
                    d_21_next_: _dafny.Seq
                    out9_: _dafny.Seq
                    out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_21_next_ = out9_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_21_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_22_ag_: _dafny.Seq
                        d_23_ai_: bool
                        d_24_ac_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                        d_22_ag_ = out10_
                        d_23_ai_ = out11_
                        d_24_ac_ = out12_
                        generated = d_22_ag_
                        insideConstrainedOut = d_23_ai_
                        currentConstrainedOut = d_24_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_25_closeBudget_: int
            d_25_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_26_cg_: _dafny.Seq
            d_27_ci_: bool
            d_28_cc_: _dafny.Seq
            out13_: _dafny.Seq
            out14_: bool
            out15_: _dafny.Seq
            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget_)
            d_26_cg_ = out13_
            d_27_ci_ = out14_
            d_28_cc_ = out15_
            generated = d_26_cg_
            insideConstrainedOut = d_27_ci_
            currentConstrainedOut = d_28_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

