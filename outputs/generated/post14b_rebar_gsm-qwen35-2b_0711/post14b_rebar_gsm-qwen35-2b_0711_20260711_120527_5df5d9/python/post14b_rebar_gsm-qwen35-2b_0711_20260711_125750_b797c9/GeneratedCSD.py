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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. At the very end of your solution, write exactly: The answer is << then the arithmetic expression using variable names and operators (+, -, *, /), then >>. Use only plain variable names from the problem statement. Do not write complex expressions or LaTeX inside the << >>. Just the simple arithmetic formula."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_phase1Budget_: int
        d_4_phase1Budget_ = 700
        if (d_4_phase1Budget_) > (maxSteps):
            d_4_phase1Budget_ = maxSteps
        with _dafny.label("0"):
            while (((d_2_steps_) < (d_4_phase1Budget_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_5_chunkBudget_: int
                    d_5_chunkBudget_ = 30
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
                        d_13_innerSteps_: int
                        d_13_innerSteps_ = 0
                        d_14_innerBudget_: int
                        d_14_innerBudget_ = 100
                        with _dafny.label("1_3_0"):
                            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_13_innerSteps_) < (d_14_innerBudget_)):
                                with _dafny.c_label("1_3_0"):
                                    if (parser).IsCompletePrefix(currentConstrainedOut):
                                        d_15_cg2_: _dafny.Seq
                                        d_16_ci2_: bool
                                        d_17_cc2_: _dafny.Seq
                                        d_18_closed2_: bool
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                        d_15_cg2_ = out7_
                                        d_16_ci2_ = out8_
                                        d_17_cc2_ = out9_
                                        d_18_closed2_ = out10_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        d_13_innerSteps_ = (d_13_innerSteps_) + (1)
                                        generated = d_15_cg2_
                                        insideConstrainedOut = d_16_ci2_
                                        currentConstrainedOut = d_17_cc2_
                                        if d_18_closed2_:
                                            d_3_hasCompletedSpan_ = True
                                    elif True:
                                        d_19_constrainedPrompt_: _dafny.Seq
                                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_20_next_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                                        d_20_next_ = out11_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        d_13_innerSteps_ = (d_13_innerSteps_) + (1)
                                        if (d_20_next_) == (eosToken):
                                            raise _dafny.Break("1_3_0")
                                        elif True:
                                            d_21_ag_: _dafny.Seq
                                            d_22_ai_: bool
                                            d_23_ac_: _dafny.Seq
                                            out12_: _dafny.Seq
                                            out13_: bool
                                            out14_: _dafny.Seq
                                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                            d_21_ag_ = out12_
                                            d_22_ai_ = out13_
                                            d_23_ac_ = out14_
                                            generated = d_21_ag_
                                            insideConstrainedOut = d_22_ai_
                                            currentConstrainedOut = d_23_ac_
                                    pass
                            pass
                        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                            d_24_closeBudget_: int
                            d_24_closeBudget_ = 60
                            d_25_remaining_: int
                            d_25_remaining_ = (maxSteps) - (d_2_steps_)
                            if (d_24_closeBudget_) > (d_25_remaining_):
                                d_24_closeBudget_ = d_25_remaining_
                            if (d_24_closeBudget_) > (0):
                                d_26_wg_: _dafny.Seq
                                d_27_wi_: bool
                                d_28_wc_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
                                d_26_wg_ = out15_
                                d_27_wi_ = out16_
                                d_28_wc_ = out17_
                                generated = d_26_wg_
                                insideConstrainedOut = d_27_wi_
                                currentConstrainedOut = d_28_wc_
                                d_2_steps_ = (d_2_steps_) + (d_24_closeBudget_)
                                if not(insideConstrainedOut):
                                    d_3_hasCompletedSpan_ = True
                        raise _dafny.Break("0")
                    pass
            pass
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            d_29_remainingForSpan_: int
            d_29_remainingForSpan_ = (maxSteps) - (d_2_steps_)
            if (d_29_remainingForSpan_) >= (2):
                d_30_fg_: _dafny.Seq
                d_31_fi_: bool
                d_32_fc_: _dafny.Seq
                out18_: _dafny.Seq
                out19_: bool
                out20_: _dafny.Seq
                out18_, out19_, out20_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_30_fg_ = out18_
                d_31_fi_ = out19_
                d_32_fc_ = out20_
                generated = d_30_fg_
                insideConstrainedOut = d_31_fi_
                currentConstrainedOut = d_32_fc_
                d_2_steps_ = (d_2_steps_) + (1)
                d_33_innerSteps2_: int
                d_33_innerSteps2_ = 0
                d_34_innerBudget2_: int
                d_34_innerBudget2_ = 80
                with _dafny.label("2_0_0"):
                    while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_33_innerSteps2_) < (d_34_innerBudget2_)):
                        with _dafny.c_label("2_0_0"):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_35_cg3_: _dafny.Seq
                                d_36_ci3_: bool
                                d_37_cc3_: _dafny.Seq
                                d_38_closed3_: bool
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out24_: bool
                                out21_, out22_, out23_, out24_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_35_cg3_ = out21_
                                d_36_ci3_ = out22_
                                d_37_cc3_ = out23_
                                d_38_closed3_ = out24_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_33_innerSteps2_ = (d_33_innerSteps2_) + (1)
                                generated = d_35_cg3_
                                insideConstrainedOut = d_36_ci3_
                                currentConstrainedOut = d_37_cc3_
                                if d_38_closed3_:
                                    d_3_hasCompletedSpan_ = True
                            elif True:
                                d_39_constrainedPrompt2_: _dafny.Seq
                                d_39_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_40_next2_: _dafny.Seq
                                out25_: _dafny.Seq
                                out25_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_39_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                d_40_next2_ = out25_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_33_innerSteps2_ = (d_33_innerSteps2_) + (1)
                                if (d_40_next2_) == (eosToken):
                                    raise _dafny.Break("2_0_0")
                                elif True:
                                    d_41_ag2_: _dafny.Seq
                                    d_42_ai2_: bool
                                    d_43_ac2_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out27_: bool
                                    out28_: _dafny.Seq
                                    out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_40_next2_)
                                    d_41_ag2_ = out26_
                                    d_42_ai2_ = out27_
                                    d_43_ac2_ = out28_
                                    generated = d_41_ag2_
                                    insideConstrainedOut = d_42_ai2_
                                    currentConstrainedOut = d_43_ac2_
                            pass
                    pass
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_44_closeBudget2_: int
                    d_44_closeBudget2_ = 60
                    d_45_remaining2_: int
                    d_45_remaining2_ = (maxSteps) - (d_2_steps_)
                    if (d_44_closeBudget2_) > (d_45_remaining2_):
                        d_44_closeBudget2_ = d_45_remaining2_
                    if (d_44_closeBudget2_) > (0):
                        d_46_wg2_: _dafny.Seq
                        d_47_wi2_: bool
                        d_48_wc2_: _dafny.Seq
                        out29_: _dafny.Seq
                        out30_: bool
                        out31_: _dafny.Seq
                        out29_, out30_, out31_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_44_closeBudget2_)
                        d_46_wg2_ = out29_
                        d_47_wi2_ = out30_
                        d_48_wc2_ = out31_
                        generated = d_46_wg2_
                        insideConstrainedOut = d_47_wi2_
                        currentConstrainedOut = d_48_wc2_
                        d_2_steps_ = (d_2_steps_) + (d_44_closeBudget2_)
                        if not(insideConstrainedOut):
                            d_3_hasCompletedSpan_ = True
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_49_finalBudget_: int
            d_49_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_49_finalBudget_) > (0):
                d_50_wgf_: _dafny.Seq
                d_51_wif_: bool
                d_52_wcf_: _dafny.Seq
                out32_: _dafny.Seq
                out33_: bool
                out34_: _dafny.Seq
                out32_, out33_, out34_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_49_finalBudget_)
                d_50_wgf_ = out32_
                d_51_wif_ = out33_
                d_52_wcf_ = out34_
                generated = d_50_wgf_
                insideConstrainedOut = d_51_wif_
                currentConstrainedOut = d_52_wcf_
                d_2_steps_ = (d_2_steps_) + (d_49_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

