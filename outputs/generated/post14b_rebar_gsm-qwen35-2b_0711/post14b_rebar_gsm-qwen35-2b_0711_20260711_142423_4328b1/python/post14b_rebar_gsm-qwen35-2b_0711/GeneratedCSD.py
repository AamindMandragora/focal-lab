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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. Use the exact variable names from the problem statement. Do NOT simplify or factor expressions. When you compute the final answer formula, write every term explicitly. At the very end, write the answer as: <<int(EXPRESSION)>> where EXPRESSION uses all required variables and operators. For example: <<int(n * frac1 * frac2)>> or <<int(n1 * frac1 + n2 * mult1)>> or <<int(n * bill - m * p1 - k * p2)>>. Do not factor out common terms. Write each term separately."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_phase1Budget_: int
        d_4_phase1Budget_ = 750
        if (d_4_phase1Budget_) > (maxSteps):
            d_4_phase1Budget_ = maxSteps
        with _dafny.label("0"):
            while (((d_2_steps_) < (d_4_phase1Budget_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_5_chunkBudget_: int
                    d_5_chunkBudget_ = 25
                    if ((d_2_steps_) + (d_5_chunkBudget_)) > (d_4_phase1Budget_):
                        d_5_chunkBudget_ = (d_4_phase1Budget_) - (d_2_steps_)
                    if (d_5_chunkBudget_) == (0):
                        raise _dafny.Break("0")
                    d_6_cg_: _dafny.Seq
                    d_7_stoppedOnOpen_: bool
                    d_8_stoppedOnEos_: bool
                    d_9_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_6_cg_ = out0_
                    d_7_stoppedOnOpen_ = out1_
                    d_8_stoppedOnEos_ = out2_
                    d_9_stepsUsed_ = out3_
                    generated = d_6_cg_
                    d_2_steps_ = (d_2_steps_) + (d_9_stepsUsed_)
                    if d_8_stoppedOnEos_:
                        raise _dafny.Break("0")
                    if d_7_stoppedOnOpen_:
                        d_10_eg_: _dafny.Seq
                        d_11_ei_: bool
                        d_12_ec_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_10_eg_ = out4_
                        d_11_ei_ = out5_
                        d_12_ec_ = out6_
                        generated = d_10_eg_
                        insideConstrainedOut = d_11_ei_
                        currentConstrainedOut = d_12_ec_
                        if (d_2_steps_) < (maxSteps):
                            d_13_closeBudget_: int
                            d_13_closeBudget_ = 120
                            d_14_remaining_: int
                            d_14_remaining_ = (maxSteps) - (d_2_steps_)
                            if (d_13_closeBudget_) > (d_14_remaining_):
                                d_13_closeBudget_ = d_14_remaining_
                            if (d_13_closeBudget_) > (0):
                                d_15_wg_: _dafny.Seq
                                d_16_wi_: bool
                                d_17_wc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_closeBudget_)
                                d_15_wg_ = out7_
                                d_16_wi_ = out8_
                                d_17_wc_ = out9_
                                generated = d_15_wg_
                                insideConstrainedOut = d_16_wi_
                                currentConstrainedOut = d_17_wc_
                                d_2_steps_ = (d_2_steps_) + (d_13_closeBudget_)
                                if not(insideConstrainedOut):
                                    d_3_hasCompletedSpan_ = True
                        d_18_innerSteps_: int
                        d_18_innerSteps_ = 0
                        d_19_innerBudget_: int
                        d_19_innerBudget_ = 60
                        with _dafny.label("1_3_0"):
                            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_18_innerSteps_) < (d_19_innerBudget_)):
                                with _dafny.c_label("1_3_0"):
                                    if (parser).IsCompletePrefix(currentConstrainedOut):
                                        d_20_cg2_: _dafny.Seq
                                        d_21_ci2_: bool
                                        d_22_cc2_: _dafny.Seq
                                        d_23_closed2_: bool
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                        d_20_cg2_ = out10_
                                        d_21_ci2_ = out11_
                                        d_22_cc2_ = out12_
                                        d_23_closed2_ = out13_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        d_18_innerSteps_ = (d_18_innerSteps_) + (1)
                                        generated = d_20_cg2_
                                        insideConstrainedOut = d_21_ci2_
                                        currentConstrainedOut = d_22_cc2_
                                        if d_23_closed2_:
                                            d_3_hasCompletedSpan_ = True
                                    elif True:
                                        d_24_constrainedPrompt_: _dafny.Seq
                                        d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_25_next_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, eosToken)
                                        d_25_next_ = out14_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        d_18_innerSteps_ = (d_18_innerSteps_) + (1)
                                        if (d_25_next_) == (eosToken):
                                            raise _dafny.Break("1_3_0")
                                        elif True:
                                            d_26_ag_: _dafny.Seq
                                            d_27_ai_: bool
                                            d_28_ac_: _dafny.Seq
                                            out15_: _dafny.Seq
                                            out16_: bool
                                            out17_: _dafny.Seq
                                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                            d_26_ag_ = out15_
                                            d_27_ai_ = out16_
                                            d_28_ac_ = out17_
                                            generated = d_26_ag_
                                            insideConstrainedOut = d_27_ai_
                                            currentConstrainedOut = d_28_ac_
                                    pass
                            pass
                        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                            d_29_closeBudget2_: int
                            d_29_closeBudget2_ = 40
                            d_30_remaining2_: int
                            d_30_remaining2_ = (maxSteps) - (d_2_steps_)
                            if (d_29_closeBudget2_) > (d_30_remaining2_):
                                d_29_closeBudget2_ = d_30_remaining2_
                            if (d_29_closeBudget2_) > (0):
                                d_31_wg2_: _dafny.Seq
                                d_32_wi2_: bool
                                d_33_wc2_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget2_)
                                d_31_wg2_ = out18_
                                d_32_wi2_ = out19_
                                d_33_wc2_ = out20_
                                generated = d_31_wg2_
                                insideConstrainedOut = d_32_wi2_
                                currentConstrainedOut = d_33_wc2_
                                d_2_steps_ = (d_2_steps_) + (d_29_closeBudget2_)
                                if not(insideConstrainedOut):
                                    d_3_hasCompletedSpan_ = True
                        raise _dafny.Break("0")
                    pass
            pass
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            d_34_remainingForSpan_: int
            d_34_remainingForSpan_ = (maxSteps) - (d_2_steps_)
            if (d_34_remainingForSpan_) >= (2):
                d_35_fg_: _dafny.Seq
                d_36_fi_: bool
                d_37_fc_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_35_fg_ = out21_
                d_36_fi_ = out22_
                d_37_fc_ = out23_
                generated = d_35_fg_
                insideConstrainedOut = d_36_fi_
                currentConstrainedOut = d_37_fc_
                d_2_steps_ = (d_2_steps_) + (1)
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_38_closeBudget3_: int
                    d_38_closeBudget3_ = 100
                    d_39_remaining3_: int
                    d_39_remaining3_ = (maxSteps) - (d_2_steps_)
                    if (d_38_closeBudget3_) > (d_39_remaining3_):
                        d_38_closeBudget3_ = d_39_remaining3_
                    if (d_38_closeBudget3_) > (0):
                        d_40_wg3_: _dafny.Seq
                        d_41_wi3_: bool
                        d_42_wc3_: _dafny.Seq
                        out24_: _dafny.Seq
                        out25_: bool
                        out26_: _dafny.Seq
                        out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_38_closeBudget3_)
                        d_40_wg3_ = out24_
                        d_41_wi3_ = out25_
                        d_42_wc3_ = out26_
                        generated = d_40_wg3_
                        insideConstrainedOut = d_41_wi3_
                        currentConstrainedOut = d_42_wc3_
                        d_2_steps_ = (d_2_steps_) + (d_38_closeBudget3_)
                        if not(insideConstrainedOut):
                            d_3_hasCompletedSpan_ = True
                d_43_innerSteps3_: int
                d_43_innerSteps3_ = 0
                d_44_innerBudget3_: int
                d_44_innerBudget3_ = 60
                with _dafny.label("2_0_0"):
                    while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_43_innerSteps3_) < (d_44_innerBudget3_)):
                        with _dafny.c_label("2_0_0"):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_45_cg4_: _dafny.Seq
                                d_46_ci4_: bool
                                d_47_cc4_: _dafny.Seq
                                d_48_closed4_: bool
                                out27_: _dafny.Seq
                                out28_: bool
                                out29_: _dafny.Seq
                                out30_: bool
                                out27_, out28_, out29_, out30_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_45_cg4_ = out27_
                                d_46_ci4_ = out28_
                                d_47_cc4_ = out29_
                                d_48_closed4_ = out30_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_43_innerSteps3_ = (d_43_innerSteps3_) + (1)
                                generated = d_45_cg4_
                                insideConstrainedOut = d_46_ci4_
                                currentConstrainedOut = d_47_cc4_
                                if d_48_closed4_:
                                    d_3_hasCompletedSpan_ = True
                            elif True:
                                d_49_constrainedPrompt3_: _dafny.Seq
                                d_49_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_50_next3_: _dafny.Seq
                                out31_: _dafny.Seq
                                out31_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_49_constrainedPrompt3_, currentConstrainedOut, eosToken)
                                d_50_next3_ = out31_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_43_innerSteps3_ = (d_43_innerSteps3_) + (1)
                                if (d_50_next3_) == (eosToken):
                                    raise _dafny.Break("2_0_0")
                                elif True:
                                    d_51_ag3_: _dafny.Seq
                                    d_52_ai3_: bool
                                    d_53_ac3_: _dafny.Seq
                                    out32_: _dafny.Seq
                                    out33_: bool
                                    out34_: _dafny.Seq
                                    out32_, out33_, out34_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_50_next3_)
                                    d_51_ag3_ = out32_
                                    d_52_ai3_ = out33_
                                    d_53_ac3_ = out34_
                                    generated = d_51_ag3_
                                    insideConstrainedOut = d_52_ai3_
                                    currentConstrainedOut = d_53_ac3_
                            pass
                    pass
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_54_closeBudget4_: int
                    d_54_closeBudget4_ = 40
                    d_55_remaining4_: int
                    d_55_remaining4_ = (maxSteps) - (d_2_steps_)
                    if (d_54_closeBudget4_) > (d_55_remaining4_):
                        d_54_closeBudget4_ = d_55_remaining4_
                    if (d_54_closeBudget4_) > (0):
                        d_56_wg4_: _dafny.Seq
                        d_57_wi4_: bool
                        d_58_wc4_: _dafny.Seq
                        out35_: _dafny.Seq
                        out36_: bool
                        out37_: _dafny.Seq
                        out35_, out36_, out37_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_54_closeBudget4_)
                        d_56_wg4_ = out35_
                        d_57_wi4_ = out36_
                        d_58_wc4_ = out37_
                        generated = d_56_wg4_
                        insideConstrainedOut = d_57_wi4_
                        currentConstrainedOut = d_58_wc4_
                        d_2_steps_ = (d_2_steps_) + (d_54_closeBudget4_)
                        if not(insideConstrainedOut):
                            d_3_hasCompletedSpan_ = True
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_59_finalBudget_: int
            d_59_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_59_finalBudget_) > (0):
                d_60_wgf_: _dafny.Seq
                d_61_wif_: bool
                d_62_wcf_: _dafny.Seq
                out38_: _dafny.Seq
                out39_: bool
                out40_: _dafny.Seq
                out38_, out39_, out40_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_59_finalBudget_)
                d_60_wgf_ = out38_
                d_61_wif_ = out39_
                d_62_wcf_ = out40_
                generated = d_60_wgf_
                insideConstrainedOut = d_61_wif_
                currentConstrainedOut = d_62_wcf_
                d_2_steps_ = (d_2_steps_) + (d_59_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

