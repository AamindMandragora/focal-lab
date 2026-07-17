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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the variable names given (like n, n1, n2, l, p, t, etc.). At the very end, write: The answer is << followed by an arithmetic expression using ONLY the variable names from the problem (no numbers, no curly braces, no currency symbols), then >>. The expression must use ALL relevant variables combined with +, -, *, /. Example: <<n1 * p + n2 * p>> not <<42>> or <<n>>. Never use a single number or single variable as the answer."))
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
                    d_5_chunkSize_: int
                    d_5_chunkSize_ = 25
                    if ((d_2_steps_) + (d_5_chunkSize_)) > (d_4_phase1Budget_):
                        d_5_chunkSize_ = (d_4_phase1Budget_) - (d_2_steps_)
                    if (d_5_chunkSize_) == (0):
                        raise _dafny.Break("0")
                    d_6_cg_: _dafny.Seq
                    d_7_stoppedOnOpen_: bool
                    d_8_stoppedOnEos_: bool
                    d_9_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
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
                        d_13_spanTokenCount_: int
                        d_13_spanTokenCount_ = 0
                        d_14_minSpanTokens_: int
                        d_14_minSpanTokens_ = 3
                        d_15_innerBudget_: int
                        d_15_innerBudget_ = 120
                        d_16_innerSteps_: int
                        d_16_innerSteps_ = 0
                        with _dafny.label("1_3_0"):
                            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_16_innerSteps_) < (d_15_innerBudget_)):
                                with _dafny.c_label("1_3_0"):
                                    if ((d_13_spanTokenCount_) >= (d_14_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_17_cg2_: _dafny.Seq
                                        d_18_ci2_: bool
                                        d_19_cc2_: _dafny.Seq
                                        d_20_closed2_: bool
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                        d_17_cg2_ = out7_
                                        d_18_ci2_ = out8_
                                        d_19_cc2_ = out9_
                                        d_20_closed2_ = out10_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        d_16_innerSteps_ = (d_16_innerSteps_) + (1)
                                        generated = d_17_cg2_
                                        insideConstrainedOut = d_18_ci2_
                                        currentConstrainedOut = d_19_cc2_
                                        if d_20_closed2_:
                                            d_3_hasCompletedSpan_ = True
                                    elif True:
                                        d_21_constrainedPrompt_: _dafny.Seq
                                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_22_next_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                        d_22_next_ = out11_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        d_16_innerSteps_ = (d_16_innerSteps_) + (1)
                                        if (d_22_next_) == (eosToken):
                                            raise _dafny.Break("1_3_0")
                                        elif True:
                                            d_23_ag_: _dafny.Seq
                                            d_24_ai_: bool
                                            d_25_ac_: _dafny.Seq
                                            out12_: _dafny.Seq
                                            out13_: bool
                                            out14_: _dafny.Seq
                                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                            d_23_ag_ = out12_
                                            d_24_ai_ = out13_
                                            d_25_ac_ = out14_
                                            generated = d_23_ag_
                                            insideConstrainedOut = d_24_ai_
                                            currentConstrainedOut = d_25_ac_
                                            d_13_spanTokenCount_ = (d_13_spanTokenCount_) + (1)
                                    pass
                            pass
                        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                            d_26_closeBudget_: int
                            d_26_closeBudget_ = 80
                            d_27_remaining_: int
                            d_27_remaining_ = (maxSteps) - (d_2_steps_)
                            if (d_26_closeBudget_) > (d_27_remaining_):
                                d_26_closeBudget_ = d_27_remaining_
                            if (d_26_closeBudget_) > (0):
                                d_28_wg_: _dafny.Seq
                                d_29_wi_: bool
                                d_30_wc_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
                                d_28_wg_ = out15_
                                d_29_wi_ = out16_
                                d_30_wc_ = out17_
                                generated = d_28_wg_
                                insideConstrainedOut = d_29_wi_
                                currentConstrainedOut = d_30_wc_
                                d_2_steps_ = (d_2_steps_) + (d_26_closeBudget_)
                                if not(insideConstrainedOut):
                                    d_3_hasCompletedSpan_ = True
                        raise _dafny.Break("0")
                    pass
            pass
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            d_31_remaining_: int
            d_31_remaining_ = (maxSteps) - (d_2_steps_)
            if (d_31_remaining_) >= (2):
                d_32_fg_: _dafny.Seq
                d_33_fi_: bool
                d_34_fc_: _dafny.Seq
                out18_: _dafny.Seq
                out19_: bool
                out20_: _dafny.Seq
                out18_, out19_, out20_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_32_fg_ = out18_
                d_33_fi_ = out19_
                d_34_fc_ = out20_
                generated = d_32_fg_
                insideConstrainedOut = d_33_fi_
                currentConstrainedOut = d_34_fc_
                d_2_steps_ = (d_2_steps_) + (1)
                d_35_spanTokenCount2_: int
                d_35_spanTokenCount2_ = 0
                d_36_minSpanTokens2_: int
                d_36_minSpanTokens2_ = 3
                d_37_innerBudget2_: int
                d_37_innerBudget2_ = 100
                d_38_innerSteps2_: int
                d_38_innerSteps2_ = 0
                with _dafny.label("2_0_0"):
                    while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_38_innerSteps2_) < (d_37_innerBudget2_)):
                        with _dafny.c_label("2_0_0"):
                            if ((d_35_spanTokenCount2_) >= (d_36_minSpanTokens2_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_39_cg3_: _dafny.Seq
                                d_40_ci3_: bool
                                d_41_cc3_: _dafny.Seq
                                d_42_closed3_: bool
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out24_: bool
                                out21_, out22_, out23_, out24_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_39_cg3_ = out21_
                                d_40_ci3_ = out22_
                                d_41_cc3_ = out23_
                                d_42_closed3_ = out24_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_38_innerSteps2_ = (d_38_innerSteps2_) + (1)
                                generated = d_39_cg3_
                                insideConstrainedOut = d_40_ci3_
                                currentConstrainedOut = d_41_cc3_
                                if d_42_closed3_:
                                    d_3_hasCompletedSpan_ = True
                            elif True:
                                d_43_constrainedPrompt2_: _dafny.Seq
                                d_43_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_44_next2_: _dafny.Seq
                                out25_: _dafny.Seq
                                out25_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_43_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_44_next2_ = out25_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_38_innerSteps2_ = (d_38_innerSteps2_) + (1)
                                if (d_44_next2_) == (eosToken):
                                    raise _dafny.Break("2_0_0")
                                elif True:
                                    d_45_ag2_: _dafny.Seq
                                    d_46_ai2_: bool
                                    d_47_ac2_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out27_: bool
                                    out28_: _dafny.Seq
                                    out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_44_next2_)
                                    d_45_ag2_ = out26_
                                    d_46_ai2_ = out27_
                                    d_47_ac2_ = out28_
                                    generated = d_45_ag2_
                                    insideConstrainedOut = d_46_ai2_
                                    currentConstrainedOut = d_47_ac2_
                                    d_35_spanTokenCount2_ = (d_35_spanTokenCount2_) + (1)
                            pass
                    pass
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_48_closeBudget3_: int
                    d_48_closeBudget3_ = 60
                    d_49_remaining3_: int
                    d_49_remaining3_ = (maxSteps) - (d_2_steps_)
                    if (d_48_closeBudget3_) > (d_49_remaining3_):
                        d_48_closeBudget3_ = d_49_remaining3_
                    if (d_48_closeBudget3_) > (0):
                        d_50_wg3_: _dafny.Seq
                        d_51_wi3_: bool
                        d_52_wc3_: _dafny.Seq
                        out29_: _dafny.Seq
                        out30_: bool
                        out31_: _dafny.Seq
                        out29_, out30_, out31_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_48_closeBudget3_)
                        d_50_wg3_ = out29_
                        d_51_wi3_ = out30_
                        d_52_wc3_ = out31_
                        generated = d_50_wg3_
                        insideConstrainedOut = d_51_wi3_
                        currentConstrainedOut = d_52_wc3_
                        d_2_steps_ = (d_2_steps_) + (d_48_closeBudget3_)
                        if not(insideConstrainedOut):
                            d_3_hasCompletedSpan_ = True
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_53_finalBudget_: int
            d_53_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_53_finalBudget_) > (0):
                d_54_wgf_: _dafny.Seq
                d_55_wif_: bool
                d_56_wcf_: _dafny.Seq
                out32_: _dafny.Seq
                out33_: bool
                out34_: _dafny.Seq
                out32_, out33_, out34_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_53_finalBudget_)
                d_54_wgf_ = out32_
                d_55_wif_ = out33_
                d_56_wcf_ = out34_
                generated = d_54_wgf_
                insideConstrainedOut = d_55_wif_
                currentConstrainedOut = d_56_wcf_
                d_2_steps_ = (d_2_steps_) + (d_53_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

