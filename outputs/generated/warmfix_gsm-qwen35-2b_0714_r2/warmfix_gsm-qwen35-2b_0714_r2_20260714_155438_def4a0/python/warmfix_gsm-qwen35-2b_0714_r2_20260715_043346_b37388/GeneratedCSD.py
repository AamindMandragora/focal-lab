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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this problem step by step. End your response with exactly: The answer is <<EXPR>> where EXPR uses variable names, numbers, and operators +, -, *, /, (, ), // only. No LaTeX, no braces {}, no ** or ^. Example: The answer is <<n * price - discount>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_prefixLimit_: int
        d_4_prefixLimit_ = 150
        if (d_4_prefixLimit_) > (maxSteps):
            d_4_prefixLimit_ = maxSteps
        d_5_chunkSize_: int
        d_5_chunkSize_ = 30
        with _dafny.label("0"):
            while (((d_2_steps_) < (d_4_prefixLimit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_6_actualChunk_: int
                    d_6_actualChunk_ = d_5_chunkSize_
                    if ((d_2_steps_) + (d_6_actualChunk_)) > (d_4_prefixLimit_):
                        d_6_actualChunk_ = (d_4_prefixLimit_) - (d_2_steps_)
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
                        raise _dafny.Break("0")
                    pass
            pass
        d_14_innerLimit_: int
        d_14_innerLimit_ = 80
        d_15_innerSteps_: int
        d_15_innerSteps_ = 0
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_15_innerSteps_) < (d_14_innerLimit_)):
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
            d_25_spanBudget_ = (maxSteps) - (d_2_steps_)
            if (d_25_spanBudget_) > (60):
                d_25_spanBudget_ = 60
            if (d_25_spanBudget_) > (0):
                d_26_wg_: _dafny.Seq
                d_27_wi_: bool
                d_28_wc_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_spanBudget_)
                d_26_wg_ = out15_
                d_27_wi_ = out16_
                d_28_wc_ = out17_
                generated = d_26_wg_
                insideConstrainedOut = d_27_wi_
                currentConstrainedOut = d_28_wc_
                d_2_steps_ = (d_2_steps_) + (d_25_spanBudget_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            d_29_freeLimit_: int
            d_29_freeLimit_ = 100
            if ((d_2_steps_) + (d_29_freeLimit_)) > (maxSteps):
                d_29_freeLimit_ = (maxSteps) - (d_2_steps_)
            with _dafny.label("5_0"):
                while ((((d_2_steps_) < ((d_2_steps_) + (d_29_freeLimit_))) and ((d_2_steps_) < (maxSteps))) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                    with _dafny.c_label("5_0"):
                        raise _dafny.Break("5_0")
                        pass
                pass
            if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
                d_30_freeChunk_: int
                d_30_freeChunk_ = 80
                if ((d_2_steps_) + (d_30_freeChunk_)) > (maxSteps):
                    d_30_freeChunk_ = (maxSteps) - (d_2_steps_)
                if (d_30_freeChunk_) > (0):
                    d_31_cg2_: _dafny.Seq
                    d_32_stoppedOnOpen2_: bool
                    d_33_stoppedOnEos2_: bool
                    d_34_stepsUsed2_: int
                    out18_: _dafny.Seq
                    out19_: bool
                    out20_: bool
                    out21_: int
                    out18_, out19_, out20_, out21_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_30_freeChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_31_cg2_ = out18_
                    d_32_stoppedOnOpen2_ = out19_
                    d_33_stoppedOnEos2_ = out20_
                    d_34_stepsUsed2_ = out21_
                    generated = d_31_cg2_
                    d_2_steps_ = (d_2_steps_) + (d_34_stepsUsed2_)
                    if d_32_stoppedOnOpen2_:
                        d_35_eg2_: _dafny.Seq
                        d_36_ei2_: bool
                        d_37_ec2_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_35_eg2_ = out22_
                        d_36_ei2_ = out23_
                        d_37_ec2_ = out24_
                        generated = d_35_eg2_
                        insideConstrainedOut = d_36_ei2_
                        currentConstrainedOut = d_37_ec2_
        d_38_innerSteps4_: int
        d_38_innerSteps4_ = 0
        d_39_innerLimit4_: int
        d_39_innerLimit4_ = 60
        with _dafny.label("2"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_38_innerSteps4_) < (d_39_innerLimit4_)):
                with _dafny.c_label("2"):
                    d_40_cg4_: _dafny.Seq
                    d_41_ci4_: bool
                    d_42_cc4_: _dafny.Seq
                    d_43_closed4_: bool
                    out25_: _dafny.Seq
                    out26_: bool
                    out27_: _dafny.Seq
                    out28_: bool
                    out25_, out26_, out27_, out28_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_40_cg4_ = out25_
                    d_41_ci4_ = out26_
                    d_42_cc4_ = out27_
                    d_43_closed4_ = out28_
                    if d_43_closed4_:
                        generated = d_40_cg4_
                        insideConstrainedOut = d_41_ci4_
                        currentConstrainedOut = d_42_cc4_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_hasCompletedSpan_ = True
                    elif True:
                        d_44_constrainedPrompt4_: _dafny.Seq
                        d_44_constrainedPrompt4_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_45_next4_: _dafny.Seq
                        out29_: _dafny.Seq
                        out29_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_44_constrainedPrompt4_, currentConstrainedOut, eosToken)
                        d_45_next4_ = out29_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_38_innerSteps4_ = (d_38_innerSteps4_) + (1)
                        if (d_45_next4_) == (eosToken):
                            raise _dafny.Break("2")
                        elif True:
                            d_46_ag4_: _dafny.Seq
                            d_47_ai4_: bool
                            d_48_ac4_: _dafny.Seq
                            out30_: _dafny.Seq
                            out31_: bool
                            out32_: _dafny.Seq
                            out30_, out31_, out32_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_45_next4_)
                            d_46_ag4_ = out30_
                            d_47_ai4_ = out31_
                            d_48_ac4_ = out32_
                            generated = d_46_ag4_
                            insideConstrainedOut = d_47_ai4_
                            currentConstrainedOut = d_48_ac4_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_49_spanBudget4_: int
            d_49_spanBudget4_ = (maxSteps) - (d_2_steps_)
            if (d_49_spanBudget4_) > (40):
                d_49_spanBudget4_ = 40
            if (d_49_spanBudget4_) > (0):
                d_50_wg4_: _dafny.Seq
                d_51_wi4_: bool
                d_52_wc4_: _dafny.Seq
                out33_: _dafny.Seq
                out34_: bool
                out35_: _dafny.Seq
                out33_, out34_, out35_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_49_spanBudget4_)
                d_50_wg4_ = out33_
                d_51_wi4_ = out34_
                d_52_wc4_ = out35_
                generated = d_50_wg4_
                insideConstrainedOut = d_51_wi4_
                currentConstrainedOut = d_52_wc4_
                d_2_steps_ = (d_2_steps_) + (d_49_spanBudget4_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and (((d_2_steps_) + (2)) <= (maxSteps)):
            d_53_fg_: _dafny.Seq
            d_54_fi_: bool
            d_55_fc_: _dafny.Seq
            out36_: _dafny.Seq
            out37_: bool
            out38_: _dafny.Seq
            out36_, out37_, out38_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_53_fg_ = out36_
            d_54_fi_ = out37_
            d_55_fc_ = out38_
            generated = d_53_fg_
            insideConstrainedOut = d_54_fi_
            currentConstrainedOut = d_55_fc_
            d_2_steps_ = (d_2_steps_) + (1)
            d_56_innerSteps5_: int
            d_56_innerSteps5_ = 0
            d_57_innerLimit5_: int
            d_57_innerLimit5_ = 50
            with _dafny.label("8_0"):
                while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_56_innerSteps5_) < (d_57_innerLimit5_)):
                    with _dafny.c_label("8_0"):
                        d_58_cg5_: _dafny.Seq
                        d_59_ci5_: bool
                        d_60_cc5_: _dafny.Seq
                        d_61_closed5_: bool
                        out39_: _dafny.Seq
                        out40_: bool
                        out41_: _dafny.Seq
                        out42_: bool
                        out39_, out40_, out41_, out42_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_58_cg5_ = out39_
                        d_59_ci5_ = out40_
                        d_60_cc5_ = out41_
                        d_61_closed5_ = out42_
                        if d_61_closed5_:
                            generated = d_58_cg5_
                            insideConstrainedOut = d_59_ci5_
                            currentConstrainedOut = d_60_cc5_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_hasCompletedSpan_ = True
                        elif True:
                            d_62_constrainedPrompt5_: _dafny.Seq
                            d_62_constrainedPrompt5_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_63_next5_: _dafny.Seq
                            out43_: _dafny.Seq
                            out43_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_62_constrainedPrompt5_, currentConstrainedOut, eosToken)
                            d_63_next5_ = out43_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_56_innerSteps5_ = (d_56_innerSteps5_) + (1)
                            if (d_63_next5_) == (eosToken):
                                raise _dafny.Break("8_0")
                            elif True:
                                d_64_ag5_: _dafny.Seq
                                d_65_ai5_: bool
                                d_66_ac5_: _dafny.Seq
                                out44_: _dafny.Seq
                                out45_: bool
                                out46_: _dafny.Seq
                                out44_, out45_, out46_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_63_next5_)
                                d_64_ag5_ = out44_
                                d_65_ai5_ = out45_
                                d_66_ac5_ = out46_
                                generated = d_64_ag5_
                                insideConstrainedOut = d_65_ai5_
                                currentConstrainedOut = d_66_ac5_
                        pass
                pass
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_67_remainBudget5_: int
                d_67_remainBudget5_ = (maxSteps) - (d_2_steps_)
                if (d_67_remainBudget5_) > (30):
                    d_67_remainBudget5_ = 30
                if (d_67_remainBudget5_) > (0):
                    d_68_wg5_: _dafny.Seq
                    d_69_wi5_: bool
                    d_70_wc5_: _dafny.Seq
                    out47_: _dafny.Seq
                    out48_: bool
                    out49_: _dafny.Seq
                    out47_, out48_, out49_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_67_remainBudget5_)
                    d_68_wg5_ = out47_
                    d_69_wi5_ = out48_
                    d_70_wc5_ = out49_
                    generated = d_68_wg5_
                    insideConstrainedOut = d_69_wi5_
                    currentConstrainedOut = d_70_wc5_
                    d_2_steps_ = (d_2_steps_) + (d_67_remainBudget5_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_71_finalBudget_: int
            d_71_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_71_finalBudget_) > (0):
                d_72_wgf_: _dafny.Seq
                d_73_wif_: bool
                d_74_wcf_: _dafny.Seq
                out50_: _dafny.Seq
                out51_: bool
                out52_: _dafny.Seq
                out50_, out51_, out52_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_71_finalBudget_)
                d_72_wgf_ = out50_
                d_73_wif_ = out51_
                d_74_wcf_ = out52_
                generated = d_72_wgf_
                insideConstrainedOut = d_73_wif_
                currentConstrainedOut = d_74_wcf_
                d_2_steps_ = (d_2_steps_) + (d_71_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

