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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write: The answer is <<EXPR>> where EXPR is a complete arithmetic expression using variable names from the problem with +, -, *, /, (, ) operators. The expression MUST combine multiple variables or numbers (e.g., <<n1 * price - n2 * discount>>). Never put just a single variable name inside << >>. Example: The answer is <<count * (n1 + n2 + n3)>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_chunkSize_: int
        d_4_chunkSize_ = 40
        d_5_phase1Limit_: int
        d_5_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (70), 100)
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
        d_14_innerStepLimit_ = 80
        d_15_innerSteps_: int
        d_15_innerSteps_ = 0
        d_16_minInnerSteps_: int
        d_16_minInnerSteps_ = 6
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_15_innerSteps_) < (d_14_innerStepLimit_)):
                with _dafny.c_label("1"):
                    if (d_15_innerSteps_) >= (d_16_minInnerSteps_):
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
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_22_next_ = out11_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_15_innerSteps_ = (d_15_innerSteps_) + (1)
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
                        d_26_constrainedPromptMin_: _dafny.Seq
                        d_26_constrainedPromptMin_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_27_nextMin_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_26_constrainedPromptMin_, currentConstrainedOut, eosToken)
                        d_27_nextMin_ = out15_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_15_innerSteps_ = (d_15_innerSteps_) + (1)
                        if (d_27_nextMin_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_28_agm_: _dafny.Seq
                            d_29_aim_: bool
                            d_30_acm_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_nextMin_)
                            d_28_agm_ = out16_
                            d_29_aim_ = out17_
                            d_30_acm_ = out18_
                            generated = d_28_agm_
                            insideConstrainedOut = d_29_aim_
                            currentConstrainedOut = d_30_acm_
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
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            d_36_phase4Limit_: int
            d_36_phase4Limit_ = (d_2_steps_) + (100)
            if (d_36_phase4Limit_) > (maxSteps):
                d_36_phase4Limit_ = maxSteps
            with _dafny.label("5_0"):
                while (((d_2_steps_) < (d_36_phase4Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                    with _dafny.c_label("5_0"):
                        d_37_actualChunk4_: int
                        d_37_actualChunk4_ = d_4_chunkSize_
                        if ((d_2_steps_) + (d_37_actualChunk4_)) > (d_36_phase4Limit_):
                            d_37_actualChunk4_ = (d_36_phase4Limit_) - (d_2_steps_)
                        if (d_37_actualChunk4_) == (0):
                            raise _dafny.Break("5_0")
                        d_38_cg4_: _dafny.Seq
                        d_39_stoppedOnOpen4_: bool
                        d_40_stoppedOnEos4_: bool
                        d_41_stepsUsed4_: int
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: bool
                        out25_: int
                        out22_, out23_, out24_, out25_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_37_actualChunk4_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_38_cg4_ = out22_
                        d_39_stoppedOnOpen4_ = out23_
                        d_40_stoppedOnEos4_ = out24_
                        d_41_stepsUsed4_ = out25_
                        generated = d_38_cg4_
                        d_2_steps_ = (d_2_steps_) + (d_41_stepsUsed4_)
                        if d_40_stoppedOnEos4_:
                            raise _dafny.Break("5_0")
                        if d_39_stoppedOnOpen4_:
                            d_42_eg4_: _dafny.Seq
                            d_43_ei4_: bool
                            d_44_ec4_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: bool
                            out28_: _dafny.Seq
                            out26_, out27_, out28_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_42_eg4_ = out26_
                            d_43_ei4_ = out27_
                            d_44_ec4_ = out28_
                            generated = d_42_eg4_
                            insideConstrainedOut = d_43_ei4_
                            currentConstrainedOut = d_44_ec4_
                        pass
                pass
            d_45_innerSteps4_: int
            d_45_innerSteps4_ = 0
            d_46_innerLimit4_: int
            d_46_innerLimit4_ = 80
            with _dafny.label("5_1"):
                while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_45_innerSteps4_) < (d_46_innerLimit4_)):
                    with _dafny.c_label("5_1"):
                        if (d_45_innerSteps4_) >= (d_16_minInnerSteps_):
                            d_47_cg4b_: _dafny.Seq
                            d_48_ci4b_: bool
                            d_49_cc4b_: _dafny.Seq
                            d_50_closed4b_: bool
                            out29_: _dafny.Seq
                            out30_: bool
                            out31_: _dafny.Seq
                            out32_: bool
                            out29_, out30_, out31_, out32_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_47_cg4b_ = out29_
                            d_48_ci4b_ = out30_
                            d_49_cc4b_ = out31_
                            d_50_closed4b_ = out32_
                            if d_50_closed4b_:
                                generated = d_47_cg4b_
                                insideConstrainedOut = d_48_ci4b_
                                currentConstrainedOut = d_49_cc4b_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_hasCompletedSpan_ = True
                            elif True:
                                d_51_cp4_: _dafny.Seq
                                d_51_cp4_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_52_next4_: _dafny.Seq
                                out33_: _dafny.Seq
                                out33_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_51_cp4_, currentConstrainedOut, eosToken)
                                d_52_next4_ = out33_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_45_innerSteps4_ = (d_45_innerSteps4_) + (1)
                                if (d_52_next4_) == (eosToken):
                                    raise _dafny.Break("5_1")
                                elif True:
                                    d_53_ag4_: _dafny.Seq
                                    d_54_ai4_: bool
                                    d_55_ac4_: _dafny.Seq
                                    out34_: _dafny.Seq
                                    out35_: bool
                                    out36_: _dafny.Seq
                                    out34_, out35_, out36_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_52_next4_)
                                    d_53_ag4_ = out34_
                                    d_54_ai4_ = out35_
                                    d_55_ac4_ = out36_
                                    generated = d_53_ag4_
                                    insideConstrainedOut = d_54_ai4_
                                    currentConstrainedOut = d_55_ac4_
                        elif True:
                            d_56_cp4m_: _dafny.Seq
                            d_56_cp4m_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_57_next4m_: _dafny.Seq
                            out37_: _dafny.Seq
                            out37_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_56_cp4m_, currentConstrainedOut, eosToken)
                            d_57_next4m_ = out37_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_45_innerSteps4_ = (d_45_innerSteps4_) + (1)
                            if (d_57_next4m_) == (eosToken):
                                raise _dafny.Break("5_1")
                            elif True:
                                d_58_ag4m_: _dafny.Seq
                                d_59_ai4m_: bool
                                d_60_ac4m_: _dafny.Seq
                                out38_: _dafny.Seq
                                out39_: bool
                                out40_: _dafny.Seq
                                out38_, out39_, out40_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_57_next4m_)
                                d_58_ag4m_ = out38_
                                d_59_ai4m_ = out39_
                                d_60_ac4m_ = out40_
                                generated = d_58_ag4m_
                                insideConstrainedOut = d_59_ai4m_
                                currentConstrainedOut = d_60_ac4m_
                        pass
                pass
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_61_spanBudget4_: int
                d_61_spanBudget4_ = 40
                d_62_remaining4_: int
                d_62_remaining4_ = (maxSteps) - (d_2_steps_)
                if (d_61_spanBudget4_) > (d_62_remaining4_):
                    d_61_spanBudget4_ = d_62_remaining4_
                if (d_61_spanBudget4_) > (0):
                    d_63_wg4_: _dafny.Seq
                    d_64_wi4_: bool
                    d_65_wc4_: _dafny.Seq
                    out41_: _dafny.Seq
                    out42_: bool
                    out43_: _dafny.Seq
                    out41_, out42_, out43_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_61_spanBudget4_)
                    d_63_wg4_ = out41_
                    d_64_wi4_ = out42_
                    d_65_wc4_ = out43_
                    generated = d_63_wg4_
                    insideConstrainedOut = d_64_wi4_
                    currentConstrainedOut = d_65_wc4_
                    d_2_steps_ = (d_2_steps_) + (d_61_spanBudget4_)
                    if not(insideConstrainedOut):
                        d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and (((d_2_steps_) + (2)) <= (maxSteps)):
            d_66_fg_: _dafny.Seq
            d_67_fi_: bool
            d_68_fc_: _dafny.Seq
            out44_: _dafny.Seq
            out45_: bool
            out46_: _dafny.Seq
            out44_, out45_, out46_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_66_fg_ = out44_
            d_67_fi_ = out45_
            d_68_fc_ = out46_
            generated = d_66_fg_
            insideConstrainedOut = d_67_fi_
            currentConstrainedOut = d_68_fc_
            d_2_steps_ = (d_2_steps_) + (1)
            d_69_innerSteps5_: int
            d_69_innerSteps5_ = 0
            d_70_innerLimit5_: int
            d_70_innerLimit5_ = 60
            with _dafny.label("6_0"):
                while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_69_innerSteps5_) < (d_70_innerLimit5_)):
                    with _dafny.c_label("6_0"):
                        if (d_69_innerSteps5_) >= (d_16_minInnerSteps_):
                            d_71_cg5_: _dafny.Seq
                            d_72_ci5_: bool
                            d_73_cc5_: _dafny.Seq
                            d_74_closed5_: bool
                            out47_: _dafny.Seq
                            out48_: bool
                            out49_: _dafny.Seq
                            out50_: bool
                            out47_, out48_, out49_, out50_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_71_cg5_ = out47_
                            d_72_ci5_ = out48_
                            d_73_cc5_ = out49_
                            d_74_closed5_ = out50_
                            if d_74_closed5_:
                                generated = d_71_cg5_
                                insideConstrainedOut = d_72_ci5_
                                currentConstrainedOut = d_73_cc5_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_hasCompletedSpan_ = True
                            elif True:
                                d_75_cp5_: _dafny.Seq
                                d_75_cp5_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_76_next5_: _dafny.Seq
                                out51_: _dafny.Seq
                                out51_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_75_cp5_, currentConstrainedOut, eosToken)
                                d_76_next5_ = out51_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_69_innerSteps5_ = (d_69_innerSteps5_) + (1)
                                if (d_76_next5_) == (eosToken):
                                    raise _dafny.Break("6_0")
                                elif True:
                                    d_77_ag5_: _dafny.Seq
                                    d_78_ai5_: bool
                                    d_79_ac5_: _dafny.Seq
                                    out52_: _dafny.Seq
                                    out53_: bool
                                    out54_: _dafny.Seq
                                    out52_, out53_, out54_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_76_next5_)
                                    d_77_ag5_ = out52_
                                    d_78_ai5_ = out53_
                                    d_79_ac5_ = out54_
                                    generated = d_77_ag5_
                                    insideConstrainedOut = d_78_ai5_
                                    currentConstrainedOut = d_79_ac5_
                        elif True:
                            d_80_cp5m_: _dafny.Seq
                            d_80_cp5m_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_81_next5m_: _dafny.Seq
                            out55_: _dafny.Seq
                            out55_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_80_cp5m_, currentConstrainedOut, eosToken)
                            d_81_next5m_ = out55_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_69_innerSteps5_ = (d_69_innerSteps5_) + (1)
                            if (d_81_next5m_) == (eosToken):
                                raise _dafny.Break("6_0")
                            elif True:
                                d_82_ag5m_: _dafny.Seq
                                d_83_ai5m_: bool
                                d_84_ac5m_: _dafny.Seq
                                out56_: _dafny.Seq
                                out57_: bool
                                out58_: _dafny.Seq
                                out56_, out57_, out58_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_81_next5m_)
                                d_82_ag5m_ = out56_
                                d_83_ai5m_ = out57_
                                d_84_ac5m_ = out58_
                                generated = d_82_ag5m_
                                insideConstrainedOut = d_83_ai5m_
                                currentConstrainedOut = d_84_ac5m_
                        pass
                pass
            if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                d_85_remainBudget5_: int
                d_85_remainBudget5_ = (maxSteps) - (d_2_steps_)
                if (d_85_remainBudget5_) > (40):
                    d_85_remainBudget5_ = 40
                if (d_85_remainBudget5_) > (0):
                    d_86_wg5_: _dafny.Seq
                    d_87_wi5_: bool
                    d_88_wc5_: _dafny.Seq
                    out59_: _dafny.Seq
                    out60_: bool
                    out61_: _dafny.Seq
                    out59_, out60_, out61_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_85_remainBudget5_)
                    d_86_wg5_ = out59_
                    d_87_wi5_ = out60_
                    d_88_wc5_ = out61_
                    generated = d_86_wg5_
                    insideConstrainedOut = d_87_wi5_
                    currentConstrainedOut = d_88_wc5_
                    d_2_steps_ = (d_2_steps_) + (d_85_remainBudget5_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_89_finalBudget_: int
            d_89_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_89_finalBudget_) > (0):
                d_90_wgf_: _dafny.Seq
                d_91_wif_: bool
                d_92_wcf_: _dafny.Seq
                out62_: _dafny.Seq
                out63_: bool
                out64_: _dafny.Seq
                out62_, out63_, out64_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_89_finalBudget_)
                d_90_wgf_ = out62_
                d_91_wif_ = out63_
                d_92_wcf_ = out64_
                generated = d_90_wgf_
                insideConstrainedOut = d_91_wif_
                currentConstrainedOut = d_92_wcf_
                d_2_steps_ = (d_2_steps_) + (d_89_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

