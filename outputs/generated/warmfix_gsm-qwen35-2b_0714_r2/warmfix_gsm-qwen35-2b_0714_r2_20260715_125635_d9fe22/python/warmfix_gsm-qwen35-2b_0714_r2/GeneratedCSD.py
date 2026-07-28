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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step using the symbolic variable names from the problem (like n1, n2, price, rate, etc.). At the very end, write exactly: The answer is <<EXPR>> where EXPR is an arithmetic expression using only those variable names, numbers, and operators +, -, *, /, (, ). No LaTeX, no braces {}, no $, no **. Keep the expression concise and correct. Example: The answer is <<(n1 + n2) * price / 60>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_chunkSize_: int
        d_4_chunkSize_ = 40
        d_5_phase1Limit_: int
        d_5_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (60), 100)
        if (d_5_phase1Limit_) == (0):
            d_5_phase1Limit_ = 1
        with _dafny.label("0"):
            while (((d_2_steps_) < (d_5_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_6_actualChunk_: int
                    d_6_actualChunk_ = d_4_chunkSize_
                    if ((d_2_steps_) + (d_6_actualChunk_)) > (d_5_phase1Limit_):
                        d_6_actualChunk_ = (d_5_phase1Limit_) - (d_2_steps_)
                    if (d_6_actualChunk_) == (0):
                        raise _dafny.Break("0")
                    d_7_cg_: _dafny.Seq
                    d_8_stoppedOnOpen_: bool
                    d_9_stoppedOnEos_: bool
                    d_10_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_7_cg_ = out0_
                    d_8_stoppedOnOpen_ = out1_
                    d_9_stoppedOnEos_ = out2_
                    d_10_stepsUsed_ = out3_
                    generated = d_7_cg_
                    d_2_steps_ = (d_2_steps_) + (d_10_stepsUsed_)
                    if d_9_stoppedOnEos_:
                        raise _dafny.Break("0")
                    if d_8_stoppedOnOpen_:
                        d_11_eg_: _dafny.Seq
                        d_12_ei_: bool
                        d_13_ec_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_11_eg_ = out4_
                        d_12_ei_ = out5_
                        d_13_ec_ = out6_
                        generated = d_11_eg_
                        insideConstrainedOut = d_12_ei_
                        currentConstrainedOut = d_13_ec_
                    pass
            pass
        d_14_innerStepLimit_: int
        d_14_innerStepLimit_ = 60
        d_15_innerSteps_: int
        d_15_innerSteps_ = 0
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_15_innerSteps_) < (d_14_innerStepLimit_)):
                with _dafny.c_label("1"):
                    d_16_cg_: _dafny.Seq
                    d_17_ci_: bool
                    d_18_cc_: _dafny.Seq
                    d_19_closed_: bool
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out10_: bool
                    out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_16_cg_ = out7_
                    d_17_ci_ = out8_
                    d_18_cc_ = out9_
                    d_19_closed_ = out10_
                    if d_19_closed_:
                        generated = d_16_cg_
                        insideConstrainedOut = d_17_ci_
                        currentConstrainedOut = d_18_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_hasCompletedSpan_ = True
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_21_next_ = out11_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_15_innerSteps_ = (d_15_innerSteps_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_22_ag_: _dafny.Seq
                            d_23_ai_: bool
                            d_24_ac_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_22_ag_ = out12_
                            d_23_ai_ = out13_
                            d_24_ac_ = out14_
                            generated = d_22_ag_
                            insideConstrainedOut = d_23_ai_
                            currentConstrainedOut = d_24_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_25_remaining_: int
            d_25_remaining_ = (maxSteps) - (d_2_steps_)
            d_26_spanBudget_: int
            d_26_spanBudget_ = d_25_remaining_
            if (d_26_spanBudget_) > (200):
                d_26_spanBudget_ = 200
            if (d_26_spanBudget_) > (0):
                d_27_wg_: _dafny.Seq
                d_28_wi_: bool
                d_29_wc_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_spanBudget_)
                d_27_wg_ = out15_
                d_28_wi_ = out16_
                d_29_wc_ = out17_
                generated = d_27_wg_
                insideConstrainedOut = d_28_wi_
                currentConstrainedOut = d_29_wc_
                d_2_steps_ = (d_2_steps_) + (d_26_spanBudget_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        d_30_phase4Limit_: int
        d_30_phase4Limit_ = _dafny.euclidian_division((maxSteps) * (80), 100)
        if (d_30_phase4Limit_) < (d_2_steps_):
            d_30_phase4Limit_ = d_2_steps_
        if (d_30_phase4Limit_) > (maxSteps):
            d_30_phase4Limit_ = maxSteps
        with _dafny.label("2"):
            while (((d_2_steps_) < (d_30_phase4Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("2"):
                    d_31_actualChunk4_: int
                    d_31_actualChunk4_ = d_4_chunkSize_
                    if ((d_2_steps_) + (d_31_actualChunk4_)) > (d_30_phase4Limit_):
                        d_31_actualChunk4_ = (d_30_phase4Limit_) - (d_2_steps_)
                    if (d_31_actualChunk4_) == (0):
                        raise _dafny.Break("2")
                    d_32_cg4_: _dafny.Seq
                    d_33_stoppedOnOpen4_: bool
                    d_34_stoppedOnEos4_: bool
                    d_35_stepsUsed4_: int
                    out18_: _dafny.Seq
                    out19_: bool
                    out20_: bool
                    out21_: int
                    out18_, out19_, out20_, out21_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_31_actualChunk4_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_32_cg4_ = out18_
                    d_33_stoppedOnOpen4_ = out19_
                    d_34_stoppedOnEos4_ = out20_
                    d_35_stepsUsed4_ = out21_
                    generated = d_32_cg4_
                    d_2_steps_ = (d_2_steps_) + (d_35_stepsUsed4_)
                    if d_34_stoppedOnEos4_:
                        raise _dafny.Break("2")
                    if d_33_stoppedOnOpen4_:
                        d_36_eg4_: _dafny.Seq
                        d_37_ei4_: bool
                        d_38_ec4_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_36_eg4_ = out22_
                        d_37_ei4_ = out23_
                        d_38_ec4_ = out24_
                        generated = d_36_eg4_
                        insideConstrainedOut = d_37_ei4_
                        currentConstrainedOut = d_38_ec4_
                    pass
            pass
        d_39_innerSteps2_: int
        d_39_innerSteps2_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_39_innerSteps2_) < (d_14_innerStepLimit_)):
                with _dafny.c_label("3"):
                    d_40_cg2_: _dafny.Seq
                    d_41_ci2_: bool
                    d_42_cc2_: _dafny.Seq
                    d_43_closed2_: bool
                    out25_: _dafny.Seq
                    out26_: bool
                    out27_: _dafny.Seq
                    out28_: bool
                    out25_, out26_, out27_, out28_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_40_cg2_ = out25_
                    d_41_ci2_ = out26_
                    d_42_cc2_ = out27_
                    d_43_closed2_ = out28_
                    if d_43_closed2_:
                        generated = d_40_cg2_
                        insideConstrainedOut = d_41_ci2_
                        currentConstrainedOut = d_42_cc2_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_hasCompletedSpan_ = True
                    elif True:
                        d_44_constrainedPrompt2_: _dafny.Seq
                        d_44_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_45_next2_: _dafny.Seq
                        out29_: _dafny.Seq
                        out29_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_44_constrainedPrompt2_, currentConstrainedOut, eosToken)
                        d_45_next2_ = out29_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_39_innerSteps2_ = (d_39_innerSteps2_) + (1)
                        if (d_45_next2_) == (eosToken):
                            raise _dafny.Break("3")
                        elif True:
                            d_46_ag2_: _dafny.Seq
                            d_47_ai2_: bool
                            d_48_ac2_: _dafny.Seq
                            out30_: _dafny.Seq
                            out31_: bool
                            out32_: _dafny.Seq
                            out30_, out31_, out32_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_45_next2_)
                            d_46_ag2_ = out30_
                            d_47_ai2_ = out31_
                            d_48_ac2_ = out32_
                            generated = d_46_ag2_
                            insideConstrainedOut = d_47_ai2_
                            currentConstrainedOut = d_48_ac2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_49_remaining6_: int
            d_49_remaining6_ = (maxSteps) - (d_2_steps_)
            d_50_spanBudget6_: int
            d_50_spanBudget6_ = d_49_remaining6_
            if (d_50_spanBudget6_) > (150):
                d_50_spanBudget6_ = 150
            if (d_50_spanBudget6_) > (0):
                d_51_wg6_: _dafny.Seq
                d_52_wi6_: bool
                d_53_wc6_: _dafny.Seq
                out33_: _dafny.Seq
                out34_: bool
                out35_: _dafny.Seq
                out33_, out34_, out35_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_50_spanBudget6_)
                d_51_wg6_ = out33_
                d_52_wi6_ = out34_
                d_53_wc6_ = out35_
                generated = d_51_wg6_
                insideConstrainedOut = d_52_wi6_
                currentConstrainedOut = d_53_wc6_
                d_2_steps_ = (d_2_steps_) + (d_50_spanBudget6_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and (((d_2_steps_) + (2)) <= (maxSteps)):
            d_54_fg_: _dafny.Seq
            d_55_fi_: bool
            d_56_fc_: _dafny.Seq
            out36_: _dafny.Seq
            out37_: bool
            out38_: _dafny.Seq
            out36_, out37_, out38_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_54_fg_ = out36_
            d_55_fi_ = out37_
            d_56_fc_ = out38_
            generated = d_54_fg_
            insideConstrainedOut = d_55_fi_
            currentConstrainedOut = d_56_fc_
            d_2_steps_ = (d_2_steps_) + (1)
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_57_remainBudget7_: int
                d_57_remainBudget7_ = (maxSteps) - (d_2_steps_)
                if (d_57_remainBudget7_) > (0):
                    d_58_wg7_: _dafny.Seq
                    d_59_wi7_: bool
                    d_60_wc7_: _dafny.Seq
                    out39_: _dafny.Seq
                    out40_: bool
                    out41_: _dafny.Seq
                    out39_, out40_, out41_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_57_remainBudget7_)
                    d_58_wg7_ = out39_
                    d_59_wi7_ = out40_
                    d_60_wc7_ = out41_
                    generated = d_58_wg7_
                    insideConstrainedOut = d_59_wi7_
                    currentConstrainedOut = d_60_wc7_
                    d_2_steps_ = (d_2_steps_) + (d_57_remainBudget7_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_61_finalBudget_: int
            d_61_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_61_finalBudget_) > (0):
                d_62_wgf_: _dafny.Seq
                d_63_wif_: bool
                d_64_wcf_: _dafny.Seq
                out42_: _dafny.Seq
                out43_: bool
                out44_: _dafny.Seq
                out42_, out43_, out44_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_61_finalBudget_)
                d_62_wgf_ = out42_
                d_63_wif_ = out43_
                d_64_wcf_ = out44_
                generated = d_62_wgf_
                insideConstrainedOut = d_63_wif_
                currentConstrainedOut = d_64_wcf_
                d_2_steps_ = (d_2_steps_) + (d_61_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

