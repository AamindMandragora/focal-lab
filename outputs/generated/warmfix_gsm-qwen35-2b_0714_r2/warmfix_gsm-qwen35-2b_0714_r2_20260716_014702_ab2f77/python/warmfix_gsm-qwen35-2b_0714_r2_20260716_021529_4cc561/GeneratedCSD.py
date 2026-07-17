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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. After working through the problem, write: The answer is <<EXPR>> where EXPR is a complete arithmetic formula using the symbolic variable names from the problem (without braces), literal numbers that appear in the problem, and operators +, -, *, /, (, ). The formula must represent the full computation including all relevant variables. Never use only a bare number as EXPR. Example: The answer is <<n * price - discount + tax>>. No LaTeX, no {}, no **, no $, no backslashes."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_chunkSize_: int
        d_4_chunkSize_ = 50
        d_5_phase1Limit_: int
        d_5_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (75), 100)
        if ((d_5_phase1Limit_) == (0)) and ((maxSteps) > (0)):
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
        d_14_innerStepLimit_ = 80
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
            d_25_spanBudget_: int
            d_25_spanBudget_ = 60
            d_26_remaining_: int
            d_26_remaining_ = (maxSteps) - (d_2_steps_)
            if (d_25_spanBudget_) > (d_26_remaining_):
                d_25_spanBudget_ = d_26_remaining_
            if (d_25_spanBudget_) > (0):
                d_27_wg_: _dafny.Seq
                d_28_wi_: bool
                d_29_wc_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_spanBudget_)
                d_27_wg_ = out15_
                d_28_wi_ = out16_
                d_29_wc_ = out17_
                generated = d_27_wg_
                insideConstrainedOut = d_28_wi_
                currentConstrainedOut = d_29_wc_
                d_2_steps_ = (d_2_steps_) + (d_25_spanBudget_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        with _dafny.label("2"):
            while (((d_2_steps_) < (d_5_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("2"):
                    d_30_next_: _dafny.Seq
                    out18_: _dafny.Seq
                    out18_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_30_next_ = out18_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_30_next_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_30_next_]))
                    if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_31_eg_: _dafny.Seq
                        d_32_ei_: bool
                        d_33_ec_: _dafny.Seq
                        out19_: _dafny.Seq
                        out20_: bool
                        out21_: _dafny.Seq
                        out19_, out20_, out21_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_31_eg_ = out19_
                        d_32_ei_ = out20_
                        d_33_ec_ = out21_
                        generated = d_31_eg_
                        insideConstrainedOut = d_32_ei_
                        currentConstrainedOut = d_33_ec_
                    pass
            pass
        d_34_innerSteps2_: int
        d_34_innerSteps2_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_34_innerSteps2_) < (d_14_innerStepLimit_)):
                with _dafny.c_label("3"):
                    d_35_cg2_: _dafny.Seq
                    d_36_ci2_: bool
                    d_37_cc2_: _dafny.Seq
                    d_38_closed2_: bool
                    out22_: _dafny.Seq
                    out23_: bool
                    out24_: _dafny.Seq
                    out25_: bool
                    out22_, out23_, out24_, out25_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_35_cg2_ = out22_
                    d_36_ci2_ = out23_
                    d_37_cc2_ = out24_
                    d_38_closed2_ = out25_
                    if d_38_closed2_:
                        generated = d_35_cg2_
                        insideConstrainedOut = d_36_ci2_
                        currentConstrainedOut = d_37_cc2_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_hasCompletedSpan_ = True
                    elif True:
                        d_39_constrainedPrompt2_: _dafny.Seq
                        d_39_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_40_next2_: _dafny.Seq
                        out26_: _dafny.Seq
                        out26_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_39_constrainedPrompt2_, currentConstrainedOut, eosToken)
                        d_40_next2_ = out26_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_34_innerSteps2_ = (d_34_innerSteps2_) + (1)
                        if (d_40_next2_) == (eosToken):
                            raise _dafny.Break("3")
                        elif True:
                            d_41_ag2_: _dafny.Seq
                            d_42_ai2_: bool
                            d_43_ac2_: _dafny.Seq
                            out27_: _dafny.Seq
                            out28_: bool
                            out29_: _dafny.Seq
                            out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_40_next2_)
                            d_41_ag2_ = out27_
                            d_42_ai2_ = out28_
                            d_43_ac2_ = out29_
                            generated = d_41_ag2_
                            insideConstrainedOut = d_42_ai2_
                            currentConstrainedOut = d_43_ac2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_44_spanBudget2_: int
            d_44_spanBudget2_ = 60
            d_45_remaining2_: int
            d_45_remaining2_ = (maxSteps) - (d_2_steps_)
            if (d_44_spanBudget2_) > (d_45_remaining2_):
                d_44_spanBudget2_ = d_45_remaining2_
            if (d_44_spanBudget2_) > (0):
                d_46_wg2_: _dafny.Seq
                d_47_wi2_: bool
                d_48_wc2_: _dafny.Seq
                out30_: _dafny.Seq
                out31_: bool
                out32_: _dafny.Seq
                out30_, out31_, out32_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_44_spanBudget2_)
                d_46_wg2_ = out30_
                d_47_wi2_ = out31_
                d_48_wc2_ = out32_
                generated = d_46_wg2_
                insideConstrainedOut = d_47_wi2_
                currentConstrainedOut = d_48_wc2_
                d_2_steps_ = (d_2_steps_) + (d_44_spanBudget2_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            if ((d_2_steps_) + (2)) <= (maxSteps):
                d_49_fg_: _dafny.Seq
                d_50_fi_: bool
                d_51_fc_: _dafny.Seq
                out33_: _dafny.Seq
                out34_: bool
                out35_: _dafny.Seq
                out33_, out34_, out35_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_49_fg_ = out33_
                d_50_fi_ = out34_
                d_51_fc_ = out35_
                generated = d_49_fg_
                insideConstrainedOut = d_50_fi_
                currentConstrainedOut = d_51_fc_
                d_2_steps_ = (d_2_steps_) + (1)
                d_52_innerSteps3_: int
                d_52_innerSteps3_ = 0
                d_53_innerLimit3_: int
                d_53_innerLimit3_ = 40
                with _dafny.label("7_0_0"):
                    while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_52_innerSteps3_) < (d_53_innerLimit3_)):
                        with _dafny.c_label("7_0_0"):
                            d_54_cg3_: _dafny.Seq
                            d_55_ci3_: bool
                            d_56_cc3_: _dafny.Seq
                            d_57_closed3_: bool
                            out36_: _dafny.Seq
                            out37_: bool
                            out38_: _dafny.Seq
                            out39_: bool
                            out36_, out37_, out38_, out39_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_54_cg3_ = out36_
                            d_55_ci3_ = out37_
                            d_56_cc3_ = out38_
                            d_57_closed3_ = out39_
                            if d_57_closed3_:
                                generated = d_54_cg3_
                                insideConstrainedOut = d_55_ci3_
                                currentConstrainedOut = d_56_cc3_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_hasCompletedSpan_ = True
                            elif True:
                                d_58_constrainedPrompt3_: _dafny.Seq
                                d_58_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_59_next3_: _dafny.Seq
                                out40_: _dafny.Seq
                                out40_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_58_constrainedPrompt3_, currentConstrainedOut, eosToken)
                                d_59_next3_ = out40_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_52_innerSteps3_ = (d_52_innerSteps3_) + (1)
                                if (d_59_next3_) == (eosToken):
                                    raise _dafny.Break("7_0_0")
                                elif True:
                                    d_60_ag3_: _dafny.Seq
                                    d_61_ai3_: bool
                                    d_62_ac3_: _dafny.Seq
                                    out41_: _dafny.Seq
                                    out42_: bool
                                    out43_: _dafny.Seq
                                    out41_, out42_, out43_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_59_next3_)
                                    d_60_ag3_ = out41_
                                    d_61_ai3_ = out42_
                                    d_62_ac3_ = out43_
                                    generated = d_60_ag3_
                                    insideConstrainedOut = d_61_ai3_
                                    currentConstrainedOut = d_62_ac3_
                            pass
                    pass
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_63_remainBudget_: int
                    d_63_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_63_remainBudget_) > (60):
                        d_63_remainBudget_ = 60
                    if (d_63_remainBudget_) > (0):
                        d_64_wg3_: _dafny.Seq
                        d_65_wi3_: bool
                        d_66_wc3_: _dafny.Seq
                        out44_: _dafny.Seq
                        out45_: bool
                        out46_: _dafny.Seq
                        out44_, out45_, out46_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_63_remainBudget_)
                        d_64_wg3_ = out44_
                        d_65_wi3_ = out45_
                        d_66_wc3_ = out46_
                        generated = d_64_wg3_
                        insideConstrainedOut = d_65_wi3_
                        currentConstrainedOut = d_66_wc3_
                        d_2_steps_ = (d_2_steps_) + (d_63_remainBudget_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_67_finalBudget_: int
            d_67_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_67_finalBudget_) > (0):
                d_68_wg4_: _dafny.Seq
                d_69_wi4_: bool
                d_70_wc4_: _dafny.Seq
                out47_: _dafny.Seq
                out48_: bool
                out49_: _dafny.Seq
                out47_, out48_, out49_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_67_finalBudget_)
                d_68_wg4_ = out47_
                d_69_wi4_ = out48_
                d_70_wc4_ = out49_
                generated = d_68_wg4_
                insideConstrainedOut = d_69_wi4_
                currentConstrainedOut = d_70_wc4_
                d_2_steps_ = (d_2_steps_) + (d_67_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

