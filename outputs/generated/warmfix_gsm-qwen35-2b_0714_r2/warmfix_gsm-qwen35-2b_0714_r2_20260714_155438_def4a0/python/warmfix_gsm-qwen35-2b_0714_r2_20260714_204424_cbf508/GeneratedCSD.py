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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step, showing all work. After completing your reasoning, write the FINAL answer using this EXACT format: The answer is <<EXPR>> where EXPR must be a complete arithmetic expression with multiple terms connected by operators (+, -, *, /). The expression must include the arithmetic operations that compute the answer, not just a single number or variable name. Use the exact variable names from the problem. Example: The answer is <<n * price - discount + bonus>> or <<(total * fraction) - current>> NOT <<42>> or <<n>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_chunkSize_: int
        d_4_chunkSize_ = 40
        d_5_phase1Limit_: int
        d_5_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (70), 100)
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
        d_16_spanTokenCount_: int
        d_16_spanTokenCount_ = 0
        d_17_minSpanTokens_: int
        d_17_minSpanTokens_ = 3
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_15_innerSteps_) < (d_14_innerStepLimit_)):
                with _dafny.c_label("1"):
                    if (d_16_spanTokenCount_) >= (d_17_minSpanTokens_):
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
                            d_23_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_23_next_ = out11_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_15_innerSteps_ = (d_15_innerSteps_) + (1)
                            if (d_23_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_24_ag_: _dafny.Seq
                                d_25_ai_: bool
                                d_26_ac_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_24_ag_ = out12_
                                d_25_ai_ = out13_
                                d_26_ac_ = out14_
                                generated = d_24_ag_
                                insideConstrainedOut = d_25_ai_
                                currentConstrainedOut = d_26_ac_
                                d_16_spanTokenCount_ = (d_16_spanTokenCount_) + (1)
                    elif True:
                        d_27_constrainedPrompt_: _dafny.Seq
                        d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_28_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_28_next_ = out15_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_15_innerSteps_ = (d_15_innerSteps_) + (1)
                        if (d_28_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_29_ag_: _dafny.Seq
                            d_30_ai_: bool
                            d_31_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                            d_29_ag_ = out16_
                            d_30_ai_ = out17_
                            d_31_ac_ = out18_
                            generated = d_29_ag_
                            insideConstrainedOut = d_30_ai_
                            currentConstrainedOut = d_31_ac_
                            d_16_spanTokenCount_ = (d_16_spanTokenCount_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_32_spanBudget_: int
            d_32_spanBudget_ = 50
            d_33_remaining_: int
            d_33_remaining_ = (maxSteps) - (d_2_steps_)
            if (d_32_spanBudget_) > (d_33_remaining_):
                d_32_spanBudget_ = d_33_remaining_
            if (d_32_spanBudget_) > (0):
                d_34_wg_: _dafny.Seq
                d_35_wi_: bool
                d_36_wc_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_spanBudget_)
                d_34_wg_ = out19_
                d_35_wi_ = out20_
                d_36_wc_ = out21_
                generated = d_34_wg_
                insideConstrainedOut = d_35_wi_
                currentConstrainedOut = d_36_wc_
                d_2_steps_ = (d_2_steps_) + (d_32_spanBudget_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        d_37_phase4Limit_: int
        d_37_phase4Limit_ = _dafny.euclidian_division((maxSteps) * (85), 100)
        if (d_37_phase4Limit_) < (d_5_phase1Limit_):
            d_37_phase4Limit_ = d_5_phase1Limit_
        with _dafny.label("2"):
            while (((d_2_steps_) < (d_37_phase4Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("2"):
                    d_38_next_: _dafny.Seq
                    out22_: _dafny.Seq
                    out22_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_38_next_ = out22_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_38_next_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_38_next_]))
                    if (d_38_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_39_eg_: _dafny.Seq
                        d_40_ei_: bool
                        d_41_ec_: _dafny.Seq
                        out23_: _dafny.Seq
                        out24_: bool
                        out25_: _dafny.Seq
                        out23_, out24_, out25_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_39_eg_ = out23_
                        d_40_ei_ = out24_
                        d_41_ec_ = out25_
                        generated = d_39_eg_
                        insideConstrainedOut = d_40_ei_
                        currentConstrainedOut = d_41_ec_
                    pass
            pass
        d_42_innerSteps2_: int
        d_42_innerSteps2_ = 0
        d_43_spanTokenCount2_: int
        d_43_spanTokenCount2_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_42_innerSteps2_) < (d_14_innerStepLimit_)):
                with _dafny.c_label("3"):
                    if (d_43_spanTokenCount2_) >= (d_17_minSpanTokens_):
                        d_44_cg2_: _dafny.Seq
                        d_45_ci2_: bool
                        d_46_cc2_: _dafny.Seq
                        d_47_closed2_: bool
                        out26_: _dafny.Seq
                        out27_: bool
                        out28_: _dafny.Seq
                        out29_: bool
                        out26_, out27_, out28_, out29_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_44_cg2_ = out26_
                        d_45_ci2_ = out27_
                        d_46_cc2_ = out28_
                        d_47_closed2_ = out29_
                        if d_47_closed2_:
                            generated = d_44_cg2_
                            insideConstrainedOut = d_45_ci2_
                            currentConstrainedOut = d_46_cc2_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_hasCompletedSpan_ = True
                        elif True:
                            d_48_constrainedPrompt2_: _dafny.Seq
                            d_48_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_49_next2_: _dafny.Seq
                            out30_: _dafny.Seq
                            out30_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_48_constrainedPrompt2_, currentConstrainedOut, eosToken)
                            d_49_next2_ = out30_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_42_innerSteps2_ = (d_42_innerSteps2_) + (1)
                            if (d_49_next2_) == (eosToken):
                                raise _dafny.Break("3")
                            elif True:
                                d_50_ag2_: _dafny.Seq
                                d_51_ai2_: bool
                                d_52_ac2_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: bool
                                out33_: _dafny.Seq
                                out31_, out32_, out33_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_49_next2_)
                                d_50_ag2_ = out31_
                                d_51_ai2_ = out32_
                                d_52_ac2_ = out33_
                                generated = d_50_ag2_
                                insideConstrainedOut = d_51_ai2_
                                currentConstrainedOut = d_52_ac2_
                                d_43_spanTokenCount2_ = (d_43_spanTokenCount2_) + (1)
                    elif True:
                        d_53_constrainedPrompt2_: _dafny.Seq
                        d_53_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_54_next2_: _dafny.Seq
                        out34_: _dafny.Seq
                        out34_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_53_constrainedPrompt2_, currentConstrainedOut, eosToken)
                        d_54_next2_ = out34_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_42_innerSteps2_ = (d_42_innerSteps2_) + (1)
                        if (d_54_next2_) == (eosToken):
                            raise _dafny.Break("3")
                        elif True:
                            d_55_ag2_: _dafny.Seq
                            d_56_ai2_: bool
                            d_57_ac2_: _dafny.Seq
                            out35_: _dafny.Seq
                            out36_: bool
                            out37_: _dafny.Seq
                            out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_54_next2_)
                            d_55_ag2_ = out35_
                            d_56_ai2_ = out36_
                            d_57_ac2_ = out37_
                            generated = d_55_ag2_
                            insideConstrainedOut = d_56_ai2_
                            currentConstrainedOut = d_57_ac2_
                            d_43_spanTokenCount2_ = (d_43_spanTokenCount2_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_58_spanBudget2_: int
            d_58_spanBudget2_ = 50
            d_59_remaining2_: int
            d_59_remaining2_ = (maxSteps) - (d_2_steps_)
            if (d_58_spanBudget2_) > (d_59_remaining2_):
                d_58_spanBudget2_ = d_59_remaining2_
            if (d_58_spanBudget2_) > (0):
                d_60_wg2_: _dafny.Seq
                d_61_wi2_: bool
                d_62_wc2_: _dafny.Seq
                out38_: _dafny.Seq
                out39_: bool
                out40_: _dafny.Seq
                out38_, out39_, out40_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_58_spanBudget2_)
                d_60_wg2_ = out38_
                d_61_wi2_ = out39_
                d_62_wc2_ = out40_
                generated = d_60_wg2_
                insideConstrainedOut = d_61_wi2_
                currentConstrainedOut = d_62_wc2_
                d_2_steps_ = (d_2_steps_) + (d_58_spanBudget2_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            if ((d_2_steps_) + (2)) <= (maxSteps):
                d_63_fg_: _dafny.Seq
                d_64_fi_: bool
                d_65_fc_: _dafny.Seq
                out41_: _dafny.Seq
                out42_: bool
                out43_: _dafny.Seq
                out41_, out42_, out43_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_63_fg_ = out41_
                d_64_fi_ = out42_
                d_65_fc_ = out43_
                generated = d_63_fg_
                insideConstrainedOut = d_64_fi_
                currentConstrainedOut = d_65_fc_
                d_2_steps_ = (d_2_steps_) + (1)
                d_66_innerSteps3_: int
                d_66_innerSteps3_ = 0
                d_67_innerLimit3_: int
                d_67_innerLimit3_ = 60
                d_68_spanTokenCount3_: int
                d_68_spanTokenCount3_ = 0
                with _dafny.label("8_0_0"):
                    while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_66_innerSteps3_) < (d_67_innerLimit3_)):
                        with _dafny.c_label("8_0_0"):
                            if (d_68_spanTokenCount3_) >= (d_17_minSpanTokens_):
                                d_69_cg3_: _dafny.Seq
                                d_70_ci3_: bool
                                d_71_cc3_: _dafny.Seq
                                d_72_closed3_: bool
                                out44_: _dafny.Seq
                                out45_: bool
                                out46_: _dafny.Seq
                                out47_: bool
                                out44_, out45_, out46_, out47_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_69_cg3_ = out44_
                                d_70_ci3_ = out45_
                                d_71_cc3_ = out46_
                                d_72_closed3_ = out47_
                                if d_72_closed3_:
                                    generated = d_69_cg3_
                                    insideConstrainedOut = d_70_ci3_
                                    currentConstrainedOut = d_71_cc3_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_3_hasCompletedSpan_ = True
                                elif True:
                                    d_73_constrainedPrompt3_: _dafny.Seq
                                    d_73_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_74_next3_: _dafny.Seq
                                    out48_: _dafny.Seq
                                    out48_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_73_constrainedPrompt3_, currentConstrainedOut, eosToken)
                                    d_74_next3_ = out48_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_66_innerSteps3_ = (d_66_innerSteps3_) + (1)
                                    if (d_74_next3_) == (eosToken):
                                        raise _dafny.Break("8_0_0")
                                    elif True:
                                        d_75_ag3_: _dafny.Seq
                                        d_76_ai3_: bool
                                        d_77_ac3_: _dafny.Seq
                                        out49_: _dafny.Seq
                                        out50_: bool
                                        out51_: _dafny.Seq
                                        out49_, out50_, out51_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_74_next3_)
                                        d_75_ag3_ = out49_
                                        d_76_ai3_ = out50_
                                        d_77_ac3_ = out51_
                                        generated = d_75_ag3_
                                        insideConstrainedOut = d_76_ai3_
                                        currentConstrainedOut = d_77_ac3_
                                        d_68_spanTokenCount3_ = (d_68_spanTokenCount3_) + (1)
                            elif True:
                                d_78_constrainedPrompt3_: _dafny.Seq
                                d_78_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_79_next3_: _dafny.Seq
                                out52_: _dafny.Seq
                                out52_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_78_constrainedPrompt3_, currentConstrainedOut, eosToken)
                                d_79_next3_ = out52_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_66_innerSteps3_ = (d_66_innerSteps3_) + (1)
                                if (d_79_next3_) == (eosToken):
                                    raise _dafny.Break("8_0_0")
                                elif True:
                                    d_80_ag3_: _dafny.Seq
                                    d_81_ai3_: bool
                                    d_82_ac3_: _dafny.Seq
                                    out53_: _dafny.Seq
                                    out54_: bool
                                    out55_: _dafny.Seq
                                    out53_, out54_, out55_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_79_next3_)
                                    d_80_ag3_ = out53_
                                    d_81_ai3_ = out54_
                                    d_82_ac3_ = out55_
                                    generated = d_80_ag3_
                                    insideConstrainedOut = d_81_ai3_
                                    currentConstrainedOut = d_82_ac3_
                                    d_68_spanTokenCount3_ = (d_68_spanTokenCount3_) + (1)
                            pass
                    pass
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_83_remainBudget_: int
                    d_83_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_83_remainBudget_) > (40):
                        d_83_remainBudget_ = 40
                    if (d_83_remainBudget_) > (0):
                        d_84_wg3_: _dafny.Seq
                        d_85_wi3_: bool
                        d_86_wc3_: _dafny.Seq
                        out56_: _dafny.Seq
                        out57_: bool
                        out58_: _dafny.Seq
                        out56_, out57_, out58_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_83_remainBudget_)
                        d_84_wg3_ = out56_
                        d_85_wi3_ = out57_
                        d_86_wc3_ = out58_
                        generated = d_84_wg3_
                        insideConstrainedOut = d_85_wi3_
                        currentConstrainedOut = d_86_wc3_
                        d_2_steps_ = (d_2_steps_) + (d_83_remainBudget_)
                        if not(insideConstrainedOut):
                            d_3_hasCompletedSpan_ = True
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_87_finalBudget_: int
            d_87_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_87_finalBudget_) > (0):
                d_88_wg4_: _dafny.Seq
                d_89_wi4_: bool
                d_90_wc4_: _dafny.Seq
                out59_: _dafny.Seq
                out60_: bool
                out61_: _dafny.Seq
                out59_, out60_, out61_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_87_finalBudget_)
                d_88_wg4_ = out59_
                d_89_wi4_ = out60_
                d_90_wc4_ = out61_
                generated = d_88_wg4_
                insideConstrainedOut = d_89_wi4_
                currentConstrainedOut = d_90_wc4_
                d_2_steps_ = (d_2_steps_) + (d_87_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

