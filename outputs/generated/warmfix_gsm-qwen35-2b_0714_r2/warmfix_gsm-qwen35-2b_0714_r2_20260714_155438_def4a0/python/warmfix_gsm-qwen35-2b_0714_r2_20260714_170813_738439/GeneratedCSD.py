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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step using the variable names from the problem. At the very end, write: The answer is <<EXPR>> where EXPR is a symbolic arithmetic expression using the exact variable names from the problem (like n, price, frac_1, total, etc.), numbers, and operators +, -, *, /. Do NOT compute numeric values - use variable names. Example: The answer is <<n * frac_1 * frac_2>> or <<frac1 * t + frac2 * (total - t)>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_chunkSize_: int
        d_4_chunkSize_ = 50
        d_5_phase1Limit_: int
        d_5_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (70), 100)
        if ((d_5_phase1Limit_) == (0)) and ((maxSteps) > (1)):
            d_5_phase1Limit_ = _dafny.euclidian_division(maxSteps, 2)
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
        d_14_minSpanLen_: int
        d_14_minSpanLen_ = 4
        d_15_innerStepLimit_: int
        d_15_innerStepLimit_ = 80
        d_16_innerSteps_: int
        d_16_innerSteps_ = 0
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_16_innerSteps_) < (d_15_innerStepLimit_)):
                with _dafny.c_label("1"):
                    if (len(currentConstrainedOut)) >= (d_14_minSpanLen_):
                        d_17_cg_: _dafny.Seq
                        d_18_ci_: bool
                        d_19_cc_: _dafny.Seq
                        d_20_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_17_cg_ = out7_
                        d_18_ci_ = out8_
                        d_19_cc_ = out9_
                        d_20_closed_ = out10_
                        if d_20_closed_:
                            generated = d_17_cg_
                            insideConstrainedOut = d_18_ci_
                            currentConstrainedOut = d_19_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_hasCompletedSpan_ = True
                        elif True:
                            d_21_constrainedPrompt_: _dafny.Seq
                            d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_22_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_22_next_ = out11_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_16_innerSteps_ = (d_16_innerSteps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("1")
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
                    elif True:
                        d_26_constrainedPrompt_: _dafny.Seq
                        d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_27_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                        d_27_next_ = out15_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_16_innerSteps_ = (d_16_innerSteps_) + (1)
                        if (d_27_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_28_ag_: _dafny.Seq
                            d_29_ai_: bool
                            d_30_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                            d_28_ag_ = out16_
                            d_29_ai_ = out17_
                            d_30_ac_ = out18_
                            generated = d_28_ag_
                            insideConstrainedOut = d_29_ai_
                            currentConstrainedOut = d_30_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_31_spanBudget_: int
            d_31_spanBudget_ = 40
            d_32_remaining_: int
            d_32_remaining_ = (maxSteps) - (d_2_steps_)
            if (d_31_spanBudget_) > (d_32_remaining_):
                d_31_spanBudget_ = d_32_remaining_
            if (d_31_spanBudget_) > (0):
                d_33_wg_: _dafny.Seq
                d_34_wi_: bool
                d_35_wc_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_31_spanBudget_)
                d_33_wg_ = out19_
                d_34_wi_ = out20_
                d_35_wc_ = out21_
                generated = d_33_wg_
                insideConstrainedOut = d_34_wi_
                currentConstrainedOut = d_35_wc_
                d_2_steps_ = (d_2_steps_) + (d_31_spanBudget_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        with _dafny.label("2"):
            while (((d_2_steps_) < (d_5_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("2"):
                    d_36_next_: _dafny.Seq
                    out22_: _dafny.Seq
                    out22_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_36_next_ = out22_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_36_next_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_36_next_]))
                    if (d_36_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_37_eg_: _dafny.Seq
                        d_38_ei_: bool
                        d_39_ec_: _dafny.Seq
                        out23_: _dafny.Seq
                        out24_: bool
                        out25_: _dafny.Seq
                        out23_, out24_, out25_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_37_eg_ = out23_
                        d_38_ei_ = out24_
                        d_39_ec_ = out25_
                        generated = d_37_eg_
                        insideConstrainedOut = d_38_ei_
                        currentConstrainedOut = d_39_ec_
                    pass
            pass
        d_40_innerSteps2_: int
        d_40_innerSteps2_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_40_innerSteps2_) < (d_15_innerStepLimit_)):
                with _dafny.c_label("3"):
                    if (len(currentConstrainedOut)) >= (d_14_minSpanLen_):
                        d_41_cg2_: _dafny.Seq
                        d_42_ci2_: bool
                        d_43_cc2_: _dafny.Seq
                        d_44_closed2_: bool
                        out26_: _dafny.Seq
                        out27_: bool
                        out28_: _dafny.Seq
                        out29_: bool
                        out26_, out27_, out28_, out29_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_41_cg2_ = out26_
                        d_42_ci2_ = out27_
                        d_43_cc2_ = out28_
                        d_44_closed2_ = out29_
                        if d_44_closed2_:
                            generated = d_41_cg2_
                            insideConstrainedOut = d_42_ci2_
                            currentConstrainedOut = d_43_cc2_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_hasCompletedSpan_ = True
                        elif True:
                            d_45_constrainedPrompt2_: _dafny.Seq
                            d_45_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_46_next2_: _dafny.Seq
                            out30_: _dafny.Seq
                            out30_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_45_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_46_next2_ = out30_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_40_innerSteps2_ = (d_40_innerSteps2_) + (1)
                            if (d_46_next2_) == (eosToken):
                                raise _dafny.Break("3")
                            elif True:
                                d_47_ag2_: _dafny.Seq
                                d_48_ai2_: bool
                                d_49_ac2_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: bool
                                out33_: _dafny.Seq
                                out31_, out32_, out33_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_46_next2_)
                                d_47_ag2_ = out31_
                                d_48_ai2_ = out32_
                                d_49_ac2_ = out33_
                                generated = d_47_ag2_
                                insideConstrainedOut = d_48_ai2_
                                currentConstrainedOut = d_49_ac2_
                    elif True:
                        d_50_constrainedPrompt2_: _dafny.Seq
                        d_50_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_51_next2_: _dafny.Seq
                        out34_: _dafny.Seq
                        out34_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_50_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                        d_51_next2_ = out34_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_40_innerSteps2_ = (d_40_innerSteps2_) + (1)
                        if (d_51_next2_) == (eosToken):
                            raise _dafny.Break("3")
                        elif True:
                            d_52_ag2_: _dafny.Seq
                            d_53_ai2_: bool
                            d_54_ac2_: _dafny.Seq
                            out35_: _dafny.Seq
                            out36_: bool
                            out37_: _dafny.Seq
                            out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_51_next2_)
                            d_52_ag2_ = out35_
                            d_53_ai2_ = out36_
                            d_54_ac2_ = out37_
                            generated = d_52_ag2_
                            insideConstrainedOut = d_53_ai2_
                            currentConstrainedOut = d_54_ac2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_55_spanBudget2_: int
            d_55_spanBudget2_ = 40
            d_56_remaining2_: int
            d_56_remaining2_ = (maxSteps) - (d_2_steps_)
            if (d_55_spanBudget2_) > (d_56_remaining2_):
                d_55_spanBudget2_ = d_56_remaining2_
            if (d_55_spanBudget2_) > (0):
                d_57_wg2_: _dafny.Seq
                d_58_wi2_: bool
                d_59_wc2_: _dafny.Seq
                out38_: _dafny.Seq
                out39_: bool
                out40_: _dafny.Seq
                out38_, out39_, out40_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_55_spanBudget2_)
                d_57_wg2_ = out38_
                d_58_wi2_ = out39_
                d_59_wc2_ = out40_
                generated = d_57_wg2_
                insideConstrainedOut = d_58_wi2_
                currentConstrainedOut = d_59_wc2_
                d_2_steps_ = (d_2_steps_) + (d_55_spanBudget2_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            if ((d_2_steps_) + (2)) <= (maxSteps):
                d_60_fg_: _dafny.Seq
                d_61_fi_: bool
                d_62_fc_: _dafny.Seq
                out41_: _dafny.Seq
                out42_: bool
                out43_: _dafny.Seq
                out41_, out42_, out43_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_60_fg_ = out41_
                d_61_fi_ = out42_
                d_62_fc_ = out43_
                generated = d_60_fg_
                insideConstrainedOut = d_61_fi_
                currentConstrainedOut = d_62_fc_
                d_2_steps_ = (d_2_steps_) + (1)
                d_63_innerSteps3_: int
                d_63_innerSteps3_ = 0
                d_64_innerLimit3_: int
                d_64_innerLimit3_ = 60
                with _dafny.label("7_0_0"):
                    while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_63_innerSteps3_) < (d_64_innerLimit3_)):
                        with _dafny.c_label("7_0_0"):
                            if (len(currentConstrainedOut)) >= (d_14_minSpanLen_):
                                d_65_cg3_: _dafny.Seq
                                d_66_ci3_: bool
                                d_67_cc3_: _dafny.Seq
                                d_68_closed3_: bool
                                out44_: _dafny.Seq
                                out45_: bool
                                out46_: _dafny.Seq
                                out47_: bool
                                out44_, out45_, out46_, out47_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_65_cg3_ = out44_
                                d_66_ci3_ = out45_
                                d_67_cc3_ = out46_
                                d_68_closed3_ = out47_
                                if d_68_closed3_:
                                    generated = d_65_cg3_
                                    insideConstrainedOut = d_66_ci3_
                                    currentConstrainedOut = d_67_cc3_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_3_hasCompletedSpan_ = True
                                elif True:
                                    d_69_constrainedPrompt3_: _dafny.Seq
                                    d_69_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_70_next3_: _dafny.Seq
                                    out48_: _dafny.Seq
                                    out48_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_69_constrainedPrompt3_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                    d_70_next3_ = out48_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_63_innerSteps3_ = (d_63_innerSteps3_) + (1)
                                    if (d_70_next3_) == (eosToken):
                                        raise _dafny.Break("7_0_0")
                                    elif True:
                                        d_71_ag3_: _dafny.Seq
                                        d_72_ai3_: bool
                                        d_73_ac3_: _dafny.Seq
                                        out49_: _dafny.Seq
                                        out50_: bool
                                        out51_: _dafny.Seq
                                        out49_, out50_, out51_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_70_next3_)
                                        d_71_ag3_ = out49_
                                        d_72_ai3_ = out50_
                                        d_73_ac3_ = out51_
                                        generated = d_71_ag3_
                                        insideConstrainedOut = d_72_ai3_
                                        currentConstrainedOut = d_73_ac3_
                            elif True:
                                d_74_constrainedPrompt3_: _dafny.Seq
                                d_74_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_75_next3_: _dafny.Seq
                                out52_: _dafny.Seq
                                out52_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_74_constrainedPrompt3_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_75_next3_ = out52_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_63_innerSteps3_ = (d_63_innerSteps3_) + (1)
                                if (d_75_next3_) == (eosToken):
                                    raise _dafny.Break("7_0_0")
                                elif True:
                                    d_76_ag3_: _dafny.Seq
                                    d_77_ai3_: bool
                                    d_78_ac3_: _dafny.Seq
                                    out53_: _dafny.Seq
                                    out54_: bool
                                    out55_: _dafny.Seq
                                    out53_, out54_, out55_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_75_next3_)
                                    d_76_ag3_ = out53_
                                    d_77_ai3_ = out54_
                                    d_78_ac3_ = out55_
                                    generated = d_76_ag3_
                                    insideConstrainedOut = d_77_ai3_
                                    currentConstrainedOut = d_78_ac3_
                            pass
                    pass
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_79_remainBudget_: int
                    d_79_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_79_remainBudget_) > (40):
                        d_79_remainBudget_ = 40
                    if (d_79_remainBudget_) > (0):
                        d_80_wg3_: _dafny.Seq
                        d_81_wi3_: bool
                        d_82_wc3_: _dafny.Seq
                        out56_: _dafny.Seq
                        out57_: bool
                        out58_: _dafny.Seq
                        out56_, out57_, out58_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_79_remainBudget_)
                        d_80_wg3_ = out56_
                        d_81_wi3_ = out57_
                        d_82_wc3_ = out58_
                        generated = d_80_wg3_
                        insideConstrainedOut = d_81_wi3_
                        currentConstrainedOut = d_82_wc3_
                        d_2_steps_ = (d_2_steps_) + (d_79_remainBudget_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_83_finalBudget_: int
            d_83_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_83_finalBudget_) > (0):
                d_84_wg4_: _dafny.Seq
                d_85_wi4_: bool
                d_86_wc4_: _dafny.Seq
                out59_: _dafny.Seq
                out60_: bool
                out61_: _dafny.Seq
                out59_, out60_, out61_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_83_finalBudget_)
                d_84_wg4_ = out59_
                d_85_wi4_ = out60_
                d_86_wc4_ = out61_
                generated = d_84_wg4_
                insideConstrainedOut = d_85_wi4_
                currentConstrainedOut = d_86_wc4_
                d_2_steps_ = (d_2_steps_) + (d_83_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

