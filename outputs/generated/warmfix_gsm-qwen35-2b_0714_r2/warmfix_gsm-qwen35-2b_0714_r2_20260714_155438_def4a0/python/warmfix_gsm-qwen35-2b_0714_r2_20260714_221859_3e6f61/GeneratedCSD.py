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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this step by step. Show all your work. At the very end, on the last line, write: The answer is <<EXPR>> where EXPR is the complete arithmetic expression from your LAST calculation step. Copy every variable and every operator exactly as you computed it. Do not simplify or drop any terms. Use only: variable names from the problem, integers, +, -, *, /, (, ). No LaTeX, no $, no **, no fractions like 1/2 (write as division). Example final line: The answer is <<(n - n2) * l * p * t / 60>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_chunkSize_: int
        d_4_chunkSize_ = 40
        d_5_phase1Limit_: int
        d_5_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (72), 100)
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
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_19_closed_:
                        generated = d_16_cg_
                        insideConstrainedOut = d_17_ci_
                        currentConstrainedOut = d_18_cc_
                        d_3_hasCompletedSpan_ = True
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        d_22_usedFallback_: bool
                        out11_: _dafny.Seq
                        out12_: bool
                        out11_, out12_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('3e0'), eosToken)
                        d_21_next_ = out11_
                        d_22_usedFallback_ = out12_
                        d_15_innerSteps_ = (d_15_innerSteps_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_23_ag_: _dafny.Seq
                            d_24_ai_: bool
                            d_25_ac_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_23_ag_ = out13_
                            d_24_ai_ = out14_
                            d_25_ac_ = out15_
                            generated = d_23_ag_
                            insideConstrainedOut = d_24_ai_
                            currentConstrainedOut = d_25_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_26_spanBudget_: int
            d_26_spanBudget_ = 50
            d_27_remaining_: int
            d_27_remaining_ = (maxSteps) - (d_2_steps_)
            if (d_26_spanBudget_) > (d_27_remaining_):
                d_26_spanBudget_ = d_27_remaining_
            if (d_26_spanBudget_) > (0):
                d_28_wg_: _dafny.Seq
                d_29_wi_: bool
                d_30_wc_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_spanBudget_)
                d_28_wg_ = out16_
                d_29_wi_ = out17_
                d_30_wc_ = out18_
                generated = d_28_wg_
                insideConstrainedOut = d_29_wi_
                currentConstrainedOut = d_30_wc_
                d_2_steps_ = (d_2_steps_) + (d_26_spanBudget_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        d_31_phase4Limit_: int
        d_31_phase4Limit_ = _dafny.euclidian_division((maxSteps) * (88), 100)
        if (d_31_phase4Limit_) < (d_5_phase1Limit_):
            d_31_phase4Limit_ = d_5_phase1Limit_
        with _dafny.label("2"):
            while (((d_2_steps_) < (d_31_phase4Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("2"):
                    d_32_next_: _dafny.Seq
                    out19_: _dafny.Seq
                    out19_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_32_next_ = out19_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_32_next_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_32_next_]))
                    if (d_32_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_33_eg_: _dafny.Seq
                        d_34_ei_: bool
                        d_35_ec_: _dafny.Seq
                        out20_: _dafny.Seq
                        out21_: bool
                        out22_: _dafny.Seq
                        out20_, out21_, out22_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_33_eg_ = out20_
                        d_34_ei_ = out21_
                        d_35_ec_ = out22_
                        generated = d_33_eg_
                        insideConstrainedOut = d_34_ei_
                        currentConstrainedOut = d_35_ec_
                    pass
            pass
        d_36_innerSteps2_: int
        d_36_innerSteps2_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_36_innerSteps2_) < (d_14_innerStepLimit_)):
                with _dafny.c_label("3"):
                    d_37_cg2_: _dafny.Seq
                    d_38_ci2_: bool
                    d_39_cc2_: _dafny.Seq
                    d_40_closed2_: bool
                    out23_: _dafny.Seq
                    out24_: bool
                    out25_: _dafny.Seq
                    out26_: bool
                    out23_, out24_, out25_, out26_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_37_cg2_ = out23_
                    d_38_ci2_ = out24_
                    d_39_cc2_ = out25_
                    d_40_closed2_ = out26_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_40_closed2_:
                        generated = d_37_cg2_
                        insideConstrainedOut = d_38_ci2_
                        currentConstrainedOut = d_39_cc2_
                        d_3_hasCompletedSpan_ = True
                    elif True:
                        d_41_constrainedPrompt2_: _dafny.Seq
                        d_41_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_42_next2_: _dafny.Seq
                        d_43_usedFallback2_: bool
                        out27_: _dafny.Seq
                        out28_: bool
                        out27_, out28_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_41_constrainedPrompt2_, currentConstrainedOut, _dafny.BigRational('3e0'), eosToken)
                        d_42_next2_ = out27_
                        d_43_usedFallback2_ = out28_
                        d_36_innerSteps2_ = (d_36_innerSteps2_) + (1)
                        if (d_42_next2_) == (eosToken):
                            raise _dafny.Break("3")
                        elif True:
                            d_44_ag2_: _dafny.Seq
                            d_45_ai2_: bool
                            d_46_ac2_: _dafny.Seq
                            out29_: _dafny.Seq
                            out30_: bool
                            out31_: _dafny.Seq
                            out29_, out30_, out31_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next2_)
                            d_44_ag2_ = out29_
                            d_45_ai2_ = out30_
                            d_46_ac2_ = out31_
                            generated = d_44_ag2_
                            insideConstrainedOut = d_45_ai2_
                            currentConstrainedOut = d_46_ac2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_47_spanBudget2_: int
            d_47_spanBudget2_ = 50
            d_48_remaining2_: int
            d_48_remaining2_ = (maxSteps) - (d_2_steps_)
            if (d_47_spanBudget2_) > (d_48_remaining2_):
                d_47_spanBudget2_ = d_48_remaining2_
            if (d_47_spanBudget2_) > (0):
                d_49_wg2_: _dafny.Seq
                d_50_wi2_: bool
                d_51_wc2_: _dafny.Seq
                out32_: _dafny.Seq
                out33_: bool
                out34_: _dafny.Seq
                out32_, out33_, out34_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_47_spanBudget2_)
                d_49_wg2_ = out32_
                d_50_wi2_ = out33_
                d_51_wc2_ = out34_
                generated = d_49_wg2_
                insideConstrainedOut = d_50_wi2_
                currentConstrainedOut = d_51_wc2_
                d_2_steps_ = (d_2_steps_) + (d_47_spanBudget2_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            if ((d_2_steps_) + (2)) <= (maxSteps):
                d_52_fg_: _dafny.Seq
                d_53_fi_: bool
                d_54_fc_: _dafny.Seq
                out35_: _dafny.Seq
                out36_: bool
                out37_: _dafny.Seq
                out35_, out36_, out37_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_52_fg_ = out35_
                d_53_fi_ = out36_
                d_54_fc_ = out37_
                generated = d_52_fg_
                insideConstrainedOut = d_53_fi_
                currentConstrainedOut = d_54_fc_
                d_2_steps_ = (d_2_steps_) + (1)
                d_55_innerSteps3_: int
                d_55_innerSteps3_ = 0
                d_56_innerLimit3_: int
                d_56_innerLimit3_ = 50
                with _dafny.label("8_0_0"):
                    while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_55_innerSteps3_) < (d_56_innerLimit3_)):
                        with _dafny.c_label("8_0_0"):
                            d_57_cg3_: _dafny.Seq
                            d_58_ci3_: bool
                            d_59_cc3_: _dafny.Seq
                            d_60_closed3_: bool
                            out38_: _dafny.Seq
                            out39_: bool
                            out40_: _dafny.Seq
                            out41_: bool
                            out38_, out39_, out40_, out41_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_57_cg3_ = out38_
                            d_58_ci3_ = out39_
                            d_59_cc3_ = out40_
                            d_60_closed3_ = out41_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_60_closed3_:
                                generated = d_57_cg3_
                                insideConstrainedOut = d_58_ci3_
                                currentConstrainedOut = d_59_cc3_
                                d_3_hasCompletedSpan_ = True
                            elif True:
                                d_61_constrainedPrompt3_: _dafny.Seq
                                d_61_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_62_next3_: _dafny.Seq
                                d_63_usedFallback3_: bool
                                out42_: _dafny.Seq
                                out43_: bool
                                out42_, out43_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_61_constrainedPrompt3_, currentConstrainedOut, _dafny.BigRational('3e0'), eosToken)
                                d_62_next3_ = out42_
                                d_63_usedFallback3_ = out43_
                                d_55_innerSteps3_ = (d_55_innerSteps3_) + (1)
                                if (d_62_next3_) == (eosToken):
                                    raise _dafny.Break("8_0_0")
                                elif True:
                                    d_64_ag3_: _dafny.Seq
                                    d_65_ai3_: bool
                                    d_66_ac3_: _dafny.Seq
                                    out44_: _dafny.Seq
                                    out45_: bool
                                    out46_: _dafny.Seq
                                    out44_, out45_, out46_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_62_next3_)
                                    d_64_ag3_ = out44_
                                    d_65_ai3_ = out45_
                                    d_66_ac3_ = out46_
                                    generated = d_64_ag3_
                                    insideConstrainedOut = d_65_ai3_
                                    currentConstrainedOut = d_66_ac3_
                            pass
                    pass
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_67_remainBudget_: int
                    d_67_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_67_remainBudget_) > (40):
                        d_67_remainBudget_ = 40
                    if (d_67_remainBudget_) > (0):
                        d_68_wg3_: _dafny.Seq
                        d_69_wi3_: bool
                        d_70_wc3_: _dafny.Seq
                        out47_: _dafny.Seq
                        out48_: bool
                        out49_: _dafny.Seq
                        out47_, out48_, out49_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_67_remainBudget_)
                        d_68_wg3_ = out47_
                        d_69_wi3_ = out48_
                        d_70_wc3_ = out49_
                        generated = d_68_wg3_
                        insideConstrainedOut = d_69_wi3_
                        currentConstrainedOut = d_70_wc3_
                        d_2_steps_ = (d_2_steps_) + (d_67_remainBudget_)
                        if not(insideConstrainedOut):
                            d_3_hasCompletedSpan_ = True
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_71_finalBudget_: int
            d_71_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_71_finalBudget_) > (0):
                d_72_wg4_: _dafny.Seq
                d_73_wi4_: bool
                d_74_wc4_: _dafny.Seq
                out50_: _dafny.Seq
                out51_: bool
                out52_: _dafny.Seq
                out50_, out51_, out52_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_71_finalBudget_)
                d_72_wg4_ = out50_
                d_73_wi4_ = out51_
                d_74_wc4_ = out52_
                generated = d_72_wg4_
                insideConstrainedOut = d_73_wi4_
                currentConstrainedOut = d_74_wc4_
                d_2_steps_ = (d_2_steps_) + (d_71_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

