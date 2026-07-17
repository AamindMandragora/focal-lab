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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step showing all reasoning. At the very end write: The answer is <<EXPR>> where EXPR is the complete arithmetic expression from your final calculation step, using variable names from the problem and operators +, -, *, /. Include ALL terms needed - do not simplify away variables. No LaTeX, no fractions notation, no backslash. Example: The answer is <<n * price - m * discount + bonus>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_chunkSize_: int
        d_4_chunkSize_ = 40
        d_5_minSpanTokens_: int
        d_5_minSpanTokens_ = 5
        d_6_phase1Limit_: int
        d_6_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (70), 100)
        if ((d_6_phase1Limit_) == (0)) and ((maxSteps) > (0)):
            d_6_phase1Limit_ = 1
        with _dafny.label("0"):
            while (((d_2_steps_) < (d_6_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_7_actualChunk_: int
                    d_7_actualChunk_ = d_4_chunkSize_
                    if ((d_2_steps_) + (d_7_actualChunk_)) > (d_6_phase1Limit_):
                        d_7_actualChunk_ = (d_6_phase1Limit_) - (d_2_steps_)
                    if (d_7_actualChunk_) == (0):
                        raise _dafny.Break("0")
                    d_8_cg_: _dafny.Seq
                    d_9_stoppedOnOpen_: bool
                    d_10_stoppedOnEos_: bool
                    d_11_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_8_cg_ = out0_
                    d_9_stoppedOnOpen_ = out1_
                    d_10_stoppedOnEos_ = out2_
                    d_11_stepsUsed_ = out3_
                    generated = d_8_cg_
                    d_2_steps_ = (d_2_steps_) + (d_11_stepsUsed_)
                    if d_10_stoppedOnEos_:
                        raise _dafny.Break("0")
                    if d_9_stoppedOnOpen_:
                        d_12_eg_: _dafny.Seq
                        d_13_ei_: bool
                        d_14_ec_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_12_eg_ = out4_
                        d_13_ei_ = out5_
                        d_14_ec_ = out6_
                        generated = d_12_eg_
                        insideConstrainedOut = d_13_ei_
                        currentConstrainedOut = d_14_ec_
                    pass
            pass
        d_15_innerStepLimit_: int
        d_15_innerStepLimit_ = 80
        d_16_innerSteps_: int
        d_16_innerSteps_ = 0
        d_17_spanCount_: int
        d_17_spanCount_ = 0
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_16_innerSteps_) < (d_15_innerStepLimit_)):
                with _dafny.c_label("1"):
                    if (d_17_spanCount_) >= (d_5_minSpanTokens_):
                        d_18_cg_: _dafny.Seq
                        d_19_ci_: bool
                        d_20_cc_: _dafny.Seq
                        d_21_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_18_cg_ = out7_
                        d_19_ci_ = out8_
                        d_20_cc_ = out9_
                        d_21_closed_ = out10_
                        if d_21_closed_:
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_hasCompletedSpan_ = True
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_penaltyTokens_: _dafny.Seq
                            d_23_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                            d_24_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_23_penaltyTokens_, _dafny.BigRational('0e0'), 4, eosToken)
                            d_24_next_ = out11_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_16_innerSteps_ = (d_16_innerSteps_) + (1)
                            if (d_24_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_25_ag_: _dafny.Seq
                                d_26_ai_: bool
                                d_27_ac_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_25_ag_ = out12_
                                d_26_ai_ = out13_
                                d_27_ac_ = out14_
                                generated = d_25_ag_
                                insideConstrainedOut = d_26_ai_
                                currentConstrainedOut = d_27_ac_
                                d_17_spanCount_ = (d_17_spanCount_) + (1)
                    elif True:
                        d_28_constrainedPrompt_: _dafny.Seq
                        d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_29_penaltyTokens_: _dafny.Seq
                        d_29_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                        d_30_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_29_penaltyTokens_, _dafny.BigRational('0e0'), 4, eosToken)
                        d_30_next_ = out15_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_16_innerSteps_ = (d_16_innerSteps_) + (1)
                        if (d_30_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_31_ag_: _dafny.Seq
                            d_32_ai_: bool
                            d_33_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                            d_31_ag_ = out16_
                            d_32_ai_ = out17_
                            d_33_ac_ = out18_
                            generated = d_31_ag_
                            insideConstrainedOut = d_32_ai_
                            currentConstrainedOut = d_33_ac_
                            d_17_spanCount_ = (d_17_spanCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_34_spanBudget_: int
            d_34_spanBudget_ = 60
            d_35_remaining_: int
            d_35_remaining_ = (maxSteps) - (d_2_steps_)
            if (d_34_spanBudget_) > (d_35_remaining_):
                d_34_spanBudget_ = d_35_remaining_
            if (d_34_spanBudget_) > (0):
                d_36_wg_: _dafny.Seq
                d_37_wi_: bool
                d_38_wc_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_34_spanBudget_)
                d_36_wg_ = out19_
                d_37_wi_ = out20_
                d_38_wc_ = out21_
                generated = d_36_wg_
                insideConstrainedOut = d_37_wi_
                currentConstrainedOut = d_38_wc_
                d_2_steps_ = (d_2_steps_) + (d_34_spanBudget_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        d_39_phase4Limit_: int
        d_39_phase4Limit_ = _dafny.euclidian_division((maxSteps) * (85), 100)
        if (d_39_phase4Limit_) < (d_6_phase1Limit_):
            d_39_phase4Limit_ = d_6_phase1Limit_
        with _dafny.label("2"):
            while (((d_2_steps_) < (d_39_phase4Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("2"):
                    d_40_next_: _dafny.Seq
                    out22_: _dafny.Seq
                    out22_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_40_next_ = out22_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_40_next_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_40_next_]))
                    if (d_40_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_41_eg_: _dafny.Seq
                        d_42_ei_: bool
                        d_43_ec_: _dafny.Seq
                        out23_: _dafny.Seq
                        out24_: bool
                        out25_: _dafny.Seq
                        out23_, out24_, out25_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_41_eg_ = out23_
                        d_42_ei_ = out24_
                        d_43_ec_ = out25_
                        generated = d_41_eg_
                        insideConstrainedOut = d_42_ei_
                        currentConstrainedOut = d_43_ec_
                    pass
            pass
        d_44_innerSteps2_: int
        d_44_innerSteps2_ = 0
        d_45_spanCount2_: int
        d_45_spanCount2_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_44_innerSteps2_) < (d_15_innerStepLimit_)):
                with _dafny.c_label("3"):
                    if (d_45_spanCount2_) >= (d_5_minSpanTokens_):
                        d_46_cg2_: _dafny.Seq
                        d_47_ci2_: bool
                        d_48_cc2_: _dafny.Seq
                        d_49_closed2_: bool
                        out26_: _dafny.Seq
                        out27_: bool
                        out28_: _dafny.Seq
                        out29_: bool
                        out26_, out27_, out28_, out29_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_46_cg2_ = out26_
                        d_47_ci2_ = out27_
                        d_48_cc2_ = out28_
                        d_49_closed2_ = out29_
                        if d_49_closed2_:
                            generated = d_46_cg2_
                            insideConstrainedOut = d_47_ci2_
                            currentConstrainedOut = d_48_cc2_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_hasCompletedSpan_ = True
                        elif True:
                            d_50_constrainedPrompt2_: _dafny.Seq
                            d_50_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_51_penaltyTokens2_: _dafny.Seq
                            d_51_penaltyTokens2_ = _dafny.SeqWithoutIsStrInference([])
                            d_52_next2_: _dafny.Seq
                            out30_: _dafny.Seq
                            out30_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_50_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_51_penaltyTokens2_, _dafny.BigRational('0e0'), 4, eosToken)
                            d_52_next2_ = out30_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_44_innerSteps2_ = (d_44_innerSteps2_) + (1)
                            if (d_52_next2_) == (eosToken):
                                raise _dafny.Break("3")
                            elif True:
                                d_53_ag2_: _dafny.Seq
                                d_54_ai2_: bool
                                d_55_ac2_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: bool
                                out33_: _dafny.Seq
                                out31_, out32_, out33_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_52_next2_)
                                d_53_ag2_ = out31_
                                d_54_ai2_ = out32_
                                d_55_ac2_ = out33_
                                generated = d_53_ag2_
                                insideConstrainedOut = d_54_ai2_
                                currentConstrainedOut = d_55_ac2_
                                d_45_spanCount2_ = (d_45_spanCount2_) + (1)
                    elif True:
                        d_56_constrainedPrompt2_: _dafny.Seq
                        d_56_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_57_penaltyTokens2_: _dafny.Seq
                        d_57_penaltyTokens2_ = _dafny.SeqWithoutIsStrInference([])
                        d_58_next2_: _dafny.Seq
                        out34_: _dafny.Seq
                        out34_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_56_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_57_penaltyTokens2_, _dafny.BigRational('0e0'), 4, eosToken)
                        d_58_next2_ = out34_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_44_innerSteps2_ = (d_44_innerSteps2_) + (1)
                        if (d_58_next2_) == (eosToken):
                            raise _dafny.Break("3")
                        elif True:
                            d_59_ag2_: _dafny.Seq
                            d_60_ai2_: bool
                            d_61_ac2_: _dafny.Seq
                            out35_: _dafny.Seq
                            out36_: bool
                            out37_: _dafny.Seq
                            out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_58_next2_)
                            d_59_ag2_ = out35_
                            d_60_ai2_ = out36_
                            d_61_ac2_ = out37_
                            generated = d_59_ag2_
                            insideConstrainedOut = d_60_ai2_
                            currentConstrainedOut = d_61_ac2_
                            d_45_spanCount2_ = (d_45_spanCount2_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_62_spanBudget2_: int
            d_62_spanBudget2_ = 60
            d_63_remaining2_: int
            d_63_remaining2_ = (maxSteps) - (d_2_steps_)
            if (d_62_spanBudget2_) > (d_63_remaining2_):
                d_62_spanBudget2_ = d_63_remaining2_
            if (d_62_spanBudget2_) > (0):
                d_64_wg2_: _dafny.Seq
                d_65_wi2_: bool
                d_66_wc2_: _dafny.Seq
                out38_: _dafny.Seq
                out39_: bool
                out40_: _dafny.Seq
                out38_, out39_, out40_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_62_spanBudget2_)
                d_64_wg2_ = out38_
                d_65_wi2_ = out39_
                d_66_wc2_ = out40_
                generated = d_64_wg2_
                insideConstrainedOut = d_65_wi2_
                currentConstrainedOut = d_66_wc2_
                d_2_steps_ = (d_2_steps_) + (d_62_spanBudget2_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            if ((d_2_steps_) + (2)) <= (maxSteps):
                d_67_fg_: _dafny.Seq
                d_68_fi_: bool
                d_69_fc_: _dafny.Seq
                out41_: _dafny.Seq
                out42_: bool
                out43_: _dafny.Seq
                out41_, out42_, out43_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_67_fg_ = out41_
                d_68_fi_ = out42_
                d_69_fc_ = out43_
                generated = d_67_fg_
                insideConstrainedOut = d_68_fi_
                currentConstrainedOut = d_69_fc_
                d_2_steps_ = (d_2_steps_) + (1)
                d_70_innerSteps3_: int
                d_70_innerSteps3_ = 0
                d_71_innerLimit3_: int
                d_71_innerLimit3_ = 60
                d_72_spanCount3_: int
                d_72_spanCount3_ = 0
                with _dafny.label("8_0_0"):
                    while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_70_innerSteps3_) < (d_71_innerLimit3_)):
                        with _dafny.c_label("8_0_0"):
                            if (d_72_spanCount3_) >= (d_5_minSpanTokens_):
                                d_73_cg3_: _dafny.Seq
                                d_74_ci3_: bool
                                d_75_cc3_: _dafny.Seq
                                d_76_closed3_: bool
                                out44_: _dafny.Seq
                                out45_: bool
                                out46_: _dafny.Seq
                                out47_: bool
                                out44_, out45_, out46_, out47_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_73_cg3_ = out44_
                                d_74_ci3_ = out45_
                                d_75_cc3_ = out46_
                                d_76_closed3_ = out47_
                                if d_76_closed3_:
                                    generated = d_73_cg3_
                                    insideConstrainedOut = d_74_ci3_
                                    currentConstrainedOut = d_75_cc3_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_3_hasCompletedSpan_ = True
                                elif True:
                                    d_77_constrainedPrompt3_: _dafny.Seq
                                    d_77_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_78_penaltyTokens3_: _dafny.Seq
                                    d_78_penaltyTokens3_ = _dafny.SeqWithoutIsStrInference([])
                                    d_79_next3_: _dafny.Seq
                                    out48_: _dafny.Seq
                                    out48_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_77_constrainedPrompt3_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_78_penaltyTokens3_, _dafny.BigRational('0e0'), 4, eosToken)
                                    d_79_next3_ = out48_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_70_innerSteps3_ = (d_70_innerSteps3_) + (1)
                                    if (d_79_next3_) == (eosToken):
                                        raise _dafny.Break("8_0_0")
                                    elif True:
                                        d_80_ag3_: _dafny.Seq
                                        d_81_ai3_: bool
                                        d_82_ac3_: _dafny.Seq
                                        out49_: _dafny.Seq
                                        out50_: bool
                                        out51_: _dafny.Seq
                                        out49_, out50_, out51_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_79_next3_)
                                        d_80_ag3_ = out49_
                                        d_81_ai3_ = out50_
                                        d_82_ac3_ = out51_
                                        generated = d_80_ag3_
                                        insideConstrainedOut = d_81_ai3_
                                        currentConstrainedOut = d_82_ac3_
                                        d_72_spanCount3_ = (d_72_spanCount3_) + (1)
                            elif True:
                                d_83_constrainedPrompt3_: _dafny.Seq
                                d_83_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_84_penaltyTokens3_: _dafny.Seq
                                d_84_penaltyTokens3_ = _dafny.SeqWithoutIsStrInference([])
                                d_85_next3_: _dafny.Seq
                                out52_: _dafny.Seq
                                out52_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_83_constrainedPrompt3_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_84_penaltyTokens3_, _dafny.BigRational('0e0'), 4, eosToken)
                                d_85_next3_ = out52_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_70_innerSteps3_ = (d_70_innerSteps3_) + (1)
                                if (d_85_next3_) == (eosToken):
                                    raise _dafny.Break("8_0_0")
                                elif True:
                                    d_86_ag3_: _dafny.Seq
                                    d_87_ai3_: bool
                                    d_88_ac3_: _dafny.Seq
                                    out53_: _dafny.Seq
                                    out54_: bool
                                    out55_: _dafny.Seq
                                    out53_, out54_, out55_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_85_next3_)
                                    d_86_ag3_ = out53_
                                    d_87_ai3_ = out54_
                                    d_88_ac3_ = out55_
                                    generated = d_86_ag3_
                                    insideConstrainedOut = d_87_ai3_
                                    currentConstrainedOut = d_88_ac3_
                                    d_72_spanCount3_ = (d_72_spanCount3_) + (1)
                            pass
                    pass
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_89_remainBudget_: int
                    d_89_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_89_remainBudget_) > (50):
                        d_89_remainBudget_ = 50
                    if (d_89_remainBudget_) > (0):
                        d_90_wg3_: _dafny.Seq
                        d_91_wi3_: bool
                        d_92_wc3_: _dafny.Seq
                        out56_: _dafny.Seq
                        out57_: bool
                        out58_: _dafny.Seq
                        out56_, out57_, out58_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_89_remainBudget_)
                        d_90_wg3_ = out56_
                        d_91_wi3_ = out57_
                        d_92_wc3_ = out58_
                        generated = d_90_wg3_
                        insideConstrainedOut = d_91_wi3_
                        currentConstrainedOut = d_92_wc3_
                        d_2_steps_ = (d_2_steps_) + (d_89_remainBudget_)
                        if not(insideConstrainedOut):
                            d_3_hasCompletedSpan_ = True
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_93_finalBudget_: int
            d_93_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_93_finalBudget_) > (0):
                d_94_wg4_: _dafny.Seq
                d_95_wi4_: bool
                d_96_wc4_: _dafny.Seq
                out59_: _dafny.Seq
                out60_: bool
                out61_: _dafny.Seq
                out59_, out60_, out61_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_93_finalBudget_)
                d_94_wg4_ = out59_
                d_95_wi4_ = out60_
                d_96_wc4_ = out61_
                generated = d_94_wg4_
                insideConstrainedOut = d_95_wi4_
                currentConstrainedOut = d_96_wc4_
                d_2_steps_ = (d_2_steps_) + (d_93_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

