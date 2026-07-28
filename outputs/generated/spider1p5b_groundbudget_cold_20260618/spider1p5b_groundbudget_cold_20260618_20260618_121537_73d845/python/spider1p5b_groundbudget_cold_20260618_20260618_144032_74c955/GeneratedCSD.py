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
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate SQL query. Output format: SQL: <<your_sql_query_here>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_chunkBudget_: int
        if (maxSteps) >= (15):
            d_3_chunkBudget_ = 10
        elif True:
            if (maxSteps) >= (2):
                d_3_chunkBudget_ = (maxSteps) - (1)
            elif True:
                d_3_chunkBudget_ = maxSteps
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_4_chunkTokens_: int
            if (d_3_chunkBudget_) > (0):
                d_4_chunkTokens_ = d_3_chunkBudget_
            elif True:
                d_4_chunkTokens_ = 0
            if (d_4_chunkTokens_) > (0):
                d_5_gOut_: _dafny.Seq
                d_6_stoppedOpen_: bool
                d_7_stoppedEos_: bool
                d_8_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkTokens_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_5_gOut_ = out0_
                d_6_stoppedOpen_ = out1_
                d_7_stoppedEos_ = out2_
                d_8_stepsUsed_ = out3_
                d_2_steps_ = (d_2_steps_) + (d_8_stepsUsed_)
                generated = d_5_gOut_
                if d_7_stoppedEos_:
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_6_stoppedOpen_:
                    d_9_eg_: _dafny.Seq
                    d_10_ei_: bool
                    d_11_ec_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_9_eg_ = out4_
                    d_10_ei_ = out5_
                    d_11_ec_ = out6_
                    generated = d_9_eg_
                    insideConstrainedOut = d_10_ei_
                    currentConstrainedOut = d_11_ec_
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_12_og_: _dafny.Seq
            d_13_oi_: bool
            d_14_oc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_12_og_ = out7_
            d_13_oi_ = out8_
            d_14_oc_ = out9_
            d_2_steps_ = (d_2_steps_) + (1)
            generated = d_12_og_
            insideConstrainedOut = d_13_oi_
            currentConstrainedOut = d_14_oc_
        d_15_closeBudgetReserve_: int
        if (maxSteps) >= (50):
            d_15_closeBudgetReserve_ = 100
        elif True:
            if (maxSteps) >= (10):
                d_15_closeBudgetReserve_ = _dafny.euclidian_division(maxSteps, 2)
            elif True:
                d_15_closeBudgetReserve_ = maxSteps
        d_16_constrainedBudget_: int
        if (maxSteps) > ((d_2_steps_) + (d_15_closeBudgetReserve_)):
            d_16_constrainedBudget_ = ((maxSteps) - (d_2_steps_)) - (d_15_closeBudgetReserve_)
        elif True:
            if (maxSteps) > (d_2_steps_):
                d_16_constrainedBudget_ = (maxSteps) - (d_2_steps_)
            elif True:
                d_16_constrainedBudget_ = 0
        d_17_constrainedSteps_: int
        d_17_constrainedSteps_ = 0
        with _dafny.label("0"):
            while ((d_17_constrainedSteps_) < (d_16_constrainedBudget_)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_18_cg_: _dafny.Seq
                    d_19_ci_: bool
                    d_20_cc_: _dafny.Seq
                    d_21_closed_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_18_cg_ = out10_
                    d_19_ci_ = out11_
                    d_20_cc_ = out12_
                    d_21_closed_ = out13_
                    d_17_constrainedSteps_ = (d_17_constrainedSteps_) + (1)
                    if d_21_closed_:
                        generated = d_18_cg_
                        insideConstrainedOut = d_19_ci_
                        currentConstrainedOut = d_20_cc_
                    elif True:
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_next_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_23_next_ = out14_
                        if (d_23_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_24_valid_: bool
                            out15_: bool
                            out15_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_23_next_)
                            d_24_valid_ = out15_
                            if d_24_valid_:
                                d_25_ag_: _dafny.Seq
                                d_26_ai_: bool
                                d_27_ac_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_25_ag_ = out16_
                                d_26_ai_ = out17_
                                d_27_ac_ = out18_
                                generated = d_25_ag_
                                insideConstrainedOut = d_26_ai_
                                currentConstrainedOut = d_27_ac_
                    pass
            pass
        d_2_steps_ = (d_2_steps_) + (d_17_constrainedSteps_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_28_remainingBudget_: int
            d_28_remainingBudget_ = (maxSteps) - (d_2_steps_)
            d_29_fg_: _dafny.Seq
            d_30_fi_: bool
            d_31_fc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_remainingBudget_)
            d_29_fg_ = out19_
            d_30_fi_ = out20_
            d_31_fc_ = out21_
            generated = d_29_fg_
            insideConstrainedOut = d_30_fi_
            currentConstrainedOut = d_31_fc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

