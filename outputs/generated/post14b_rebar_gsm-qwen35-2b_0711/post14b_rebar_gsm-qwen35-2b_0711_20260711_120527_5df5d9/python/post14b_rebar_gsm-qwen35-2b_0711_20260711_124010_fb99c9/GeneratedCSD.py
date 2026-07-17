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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. At the very end, write: The answer is <<EXPR>> where EXPR is the COMPLETE arithmetic expression using ALL necessary variable names from the problem combined with operators +, -, *, /. Do NOT truncate the expression - include every variable needed. Use only plain variable names (no {}, no LaTeX, no **, no $). Example: The answer is <<total - n1 - n2>> not <<total - n1>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_minSpanTokens_: int
        d_4_minSpanTokens_ = 5
        d_5_spanTokensGenerated_: int
        d_5_spanTokensGenerated_ = 0
        d_6_phase1Limit_: int
        d_6_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (75), 100)
        if ((d_6_phase1Limit_) == (0)) and ((maxSteps) > (0)):
            d_6_phase1Limit_ = 1
        d_7_chunkSize_: int
        d_7_chunkSize_ = 50
        with _dafny.label("0"):
            while (((d_2_steps_) < (d_6_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_8_actualChunk_: int
                    d_8_actualChunk_ = d_7_chunkSize_
                    if ((d_2_steps_) + (d_8_actualChunk_)) > (d_6_phase1Limit_):
                        d_8_actualChunk_ = (d_6_phase1Limit_) - (d_2_steps_)
                    if (d_8_actualChunk_) == (0):
                        raise _dafny.Break("0")
                    d_9_cg_: _dafny.Seq
                    d_10_stoppedOnOpen_: bool
                    d_11_stoppedOnEos_: bool
                    d_12_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_9_cg_ = out0_
                    d_10_stoppedOnOpen_ = out1_
                    d_11_stoppedOnEos_ = out2_
                    d_12_stepsUsed_ = out3_
                    generated = d_9_cg_
                    d_2_steps_ = (d_2_steps_) + (d_12_stepsUsed_)
                    if d_11_stoppedOnEos_:
                        raise _dafny.Break("0")
                    if d_10_stoppedOnOpen_:
                        d_13_eg_: _dafny.Seq
                        d_14_ei_: bool
                        d_15_ec_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_13_eg_ = out4_
                        d_14_ei_ = out5_
                        d_15_ec_ = out6_
                        generated = d_13_eg_
                        insideConstrainedOut = d_14_ei_
                        currentConstrainedOut = d_15_ec_
                        d_5_spanTokensGenerated_ = 0
                    pass
            pass
        d_16_innerStepLimit_: int
        d_16_innerStepLimit_ = 80
        d_17_innerSteps_: int
        d_17_innerSteps_ = 0
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_17_innerSteps_) < (d_16_innerStepLimit_)):
                with _dafny.c_label("1"):
                    if ((d_5_spanTokensGenerated_) >= (d_4_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_18_cg2_: _dafny.Seq
                        d_19_ci2_: bool
                        d_20_cc2_: _dafny.Seq
                        d_21_closed2_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_18_cg2_ = out7_
                        d_19_ci2_ = out8_
                        d_20_cc2_ = out9_
                        d_21_closed2_ = out10_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_17_innerSteps_ = (d_17_innerSteps_) + (1)
                        if d_21_closed2_:
                            generated = d_18_cg2_
                            insideConstrainedOut = d_19_ci2_
                            currentConstrainedOut = d_20_cc2_
                            d_3_hasCompletedSpan_ = True
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                            d_23_next_ = out11_
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
                                d_5_spanTokensGenerated_ = (d_5_spanTokensGenerated_) + (1)
                    elif True:
                        d_27_constrainedPrompt_: _dafny.Seq
                        d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_28_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                        d_28_next_ = out15_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_17_innerSteps_ = (d_17_innerSteps_) + (1)
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
                            d_5_spanTokensGenerated_ = (d_5_spanTokensGenerated_) + (1)
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
        with _dafny.label("2"):
            while (((d_2_steps_) < (d_6_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("2"):
                    d_37_next_: _dafny.Seq
                    out22_: _dafny.Seq
                    out22_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_37_next_ = out22_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_37_next_) == (eosToken):
                        raise _dafny.Break("2")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_37_next_]))
                    if (d_37_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        d_38_eg_: _dafny.Seq
                        d_39_ei_: bool
                        d_40_ec_: _dafny.Seq
                        out23_: _dafny.Seq
                        out24_: bool
                        out25_: _dafny.Seq
                        out23_, out24_, out25_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_38_eg_ = out23_
                        d_39_ei_ = out24_
                        d_40_ec_ = out25_
                        generated = d_38_eg_
                        insideConstrainedOut = d_39_ei_
                        currentConstrainedOut = d_40_ec_
                        d_5_spanTokensGenerated_ = 0
                    pass
            pass
        d_41_innerSteps5_: int
        d_41_innerSteps5_ = 0
        with _dafny.label("3"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_41_innerSteps5_) < (d_16_innerStepLimit_)):
                with _dafny.c_label("3"):
                    if ((d_5_spanTokensGenerated_) >= (d_4_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_42_cg5_: _dafny.Seq
                        d_43_ci5_: bool
                        d_44_cc5_: _dafny.Seq
                        d_45_closed5_: bool
                        out26_: _dafny.Seq
                        out27_: bool
                        out28_: _dafny.Seq
                        out29_: bool
                        out26_, out27_, out28_, out29_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_42_cg5_ = out26_
                        d_43_ci5_ = out27_
                        d_44_cc5_ = out28_
                        d_45_closed5_ = out29_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_41_innerSteps5_ = (d_41_innerSteps5_) + (1)
                        if d_45_closed5_:
                            generated = d_42_cg5_
                            insideConstrainedOut = d_43_ci5_
                            currentConstrainedOut = d_44_cc5_
                            d_3_hasCompletedSpan_ = True
                        elif True:
                            d_46_constrainedPrompt5_: _dafny.Seq
                            d_46_constrainedPrompt5_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_47_next5_: _dafny.Seq
                            out30_: _dafny.Seq
                            out30_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_46_constrainedPrompt5_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                            d_47_next5_ = out30_
                            if (d_47_next5_) == (eosToken):
                                raise _dafny.Break("3")
                            elif True:
                                d_48_ag5_: _dafny.Seq
                                d_49_ai5_: bool
                                d_50_ac5_: _dafny.Seq
                                out31_: _dafny.Seq
                                out32_: bool
                                out33_: _dafny.Seq
                                out31_, out32_, out33_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_47_next5_)
                                d_48_ag5_ = out31_
                                d_49_ai5_ = out32_
                                d_50_ac5_ = out33_
                                generated = d_48_ag5_
                                insideConstrainedOut = d_49_ai5_
                                currentConstrainedOut = d_50_ac5_
                                d_5_spanTokensGenerated_ = (d_5_spanTokensGenerated_) + (1)
                    elif True:
                        d_51_constrainedPrompt5_: _dafny.Seq
                        d_51_constrainedPrompt5_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_52_next5_: _dafny.Seq
                        out34_: _dafny.Seq
                        out34_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_51_constrainedPrompt5_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                        d_52_next5_ = out34_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_41_innerSteps5_ = (d_41_innerSteps5_) + (1)
                        if (d_52_next5_) == (eosToken):
                            raise _dafny.Break("3")
                        elif True:
                            d_53_ag5_: _dafny.Seq
                            d_54_ai5_: bool
                            d_55_ac5_: _dafny.Seq
                            out35_: _dafny.Seq
                            out36_: bool
                            out37_: _dafny.Seq
                            out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_52_next5_)
                            d_53_ag5_ = out35_
                            d_54_ai5_ = out36_
                            d_55_ac5_ = out37_
                            generated = d_53_ag5_
                            insideConstrainedOut = d_54_ai5_
                            currentConstrainedOut = d_55_ac5_
                            d_5_spanTokensGenerated_ = (d_5_spanTokensGenerated_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_56_spanBudget5_: int
            d_56_spanBudget5_ = 50
            d_57_remaining5_: int
            d_57_remaining5_ = (maxSteps) - (d_2_steps_)
            if (d_56_spanBudget5_) > (d_57_remaining5_):
                d_56_spanBudget5_ = d_57_remaining5_
            if (d_56_spanBudget5_) > (0):
                d_58_wg5_: _dafny.Seq
                d_59_wi5_: bool
                d_60_wc5_: _dafny.Seq
                out38_: _dafny.Seq
                out39_: bool
                out40_: _dafny.Seq
                out38_, out39_, out40_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_56_spanBudget5_)
                d_58_wg5_ = out38_
                d_59_wi5_ = out39_
                d_60_wc5_ = out40_
                generated = d_58_wg5_
                insideConstrainedOut = d_59_wi5_
                currentConstrainedOut = d_60_wc5_
                d_2_steps_ = (d_2_steps_) + (d_56_spanBudget5_)
                if not(insideConstrainedOut):
                    d_3_hasCompletedSpan_ = True
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            if ((d_2_steps_) + (2)) <= (maxSteps):
                d_61_fg_: _dafny.Seq
                d_62_fi_: bool
                d_63_fc_: _dafny.Seq
                out41_: _dafny.Seq
                out42_: bool
                out43_: _dafny.Seq
                out41_, out42_, out43_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_61_fg_ = out41_
                d_62_fi_ = out42_
                d_63_fc_ = out43_
                generated = d_61_fg_
                insideConstrainedOut = d_62_fi_
                currentConstrainedOut = d_63_fc_
                d_2_steps_ = (d_2_steps_) + (1)
                d_5_spanTokensGenerated_ = 0
                d_64_innerSteps6_: int
                d_64_innerSteps6_ = 0
                d_65_innerLimit6_: int
                d_65_innerLimit6_ = 60
                with _dafny.label("7_0_0"):
                    while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_64_innerSteps6_) < (d_65_innerLimit6_)):
                        with _dafny.c_label("7_0_0"):
                            if ((d_5_spanTokensGenerated_) >= (d_4_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_66_cg6_: _dafny.Seq
                                d_67_ci6_: bool
                                d_68_cc6_: _dafny.Seq
                                d_69_closed6_: bool
                                out44_: _dafny.Seq
                                out45_: bool
                                out46_: _dafny.Seq
                                out47_: bool
                                out44_, out45_, out46_, out47_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_66_cg6_ = out44_
                                d_67_ci6_ = out45_
                                d_68_cc6_ = out46_
                                d_69_closed6_ = out47_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_64_innerSteps6_ = (d_64_innerSteps6_) + (1)
                                if d_69_closed6_:
                                    generated = d_66_cg6_
                                    insideConstrainedOut = d_67_ci6_
                                    currentConstrainedOut = d_68_cc6_
                                    d_3_hasCompletedSpan_ = True
                                elif True:
                                    d_70_constrainedPrompt6_: _dafny.Seq
                                    d_70_constrainedPrompt6_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_71_next6_: _dafny.Seq
                                    out48_: _dafny.Seq
                                    out48_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_70_constrainedPrompt6_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                                    d_71_next6_ = out48_
                                    if (d_71_next6_) == (eosToken):
                                        raise _dafny.Break("7_0_0")
                                    elif True:
                                        d_72_ag6_: _dafny.Seq
                                        d_73_ai6_: bool
                                        d_74_ac6_: _dafny.Seq
                                        out49_: _dafny.Seq
                                        out50_: bool
                                        out51_: _dafny.Seq
                                        out49_, out50_, out51_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_71_next6_)
                                        d_72_ag6_ = out49_
                                        d_73_ai6_ = out50_
                                        d_74_ac6_ = out51_
                                        generated = d_72_ag6_
                                        insideConstrainedOut = d_73_ai6_
                                        currentConstrainedOut = d_74_ac6_
                                        d_5_spanTokensGenerated_ = (d_5_spanTokensGenerated_) + (1)
                            elif True:
                                d_75_constrainedPrompt6_: _dafny.Seq
                                d_75_constrainedPrompt6_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_76_next6_: _dafny.Seq
                                out52_: _dafny.Seq
                                out52_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_75_constrainedPrompt6_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                                d_76_next6_ = out52_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_64_innerSteps6_ = (d_64_innerSteps6_) + (1)
                                if (d_76_next6_) == (eosToken):
                                    raise _dafny.Break("7_0_0")
                                elif True:
                                    d_77_ag6_: _dafny.Seq
                                    d_78_ai6_: bool
                                    d_79_ac6_: _dafny.Seq
                                    out53_: _dafny.Seq
                                    out54_: bool
                                    out55_: _dafny.Seq
                                    out53_, out54_, out55_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_76_next6_)
                                    d_77_ag6_ = out53_
                                    d_78_ai6_ = out54_
                                    d_79_ac6_ = out55_
                                    generated = d_77_ag6_
                                    insideConstrainedOut = d_78_ai6_
                                    currentConstrainedOut = d_79_ac6_
                                    d_5_spanTokensGenerated_ = (d_5_spanTokensGenerated_) + (1)
                            pass
                    pass
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_80_remainBudget6_: int
                    d_80_remainBudget6_ = (maxSteps) - (d_2_steps_)
                    if (d_80_remainBudget6_) > (50):
                        d_80_remainBudget6_ = 50
                    if (d_80_remainBudget6_) > (0):
                        d_81_wg6_: _dafny.Seq
                        d_82_wi6_: bool
                        d_83_wc6_: _dafny.Seq
                        out56_: _dafny.Seq
                        out57_: bool
                        out58_: _dafny.Seq
                        out56_, out57_, out58_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_80_remainBudget6_)
                        d_81_wg6_ = out56_
                        d_82_wi6_ = out57_
                        d_83_wc6_ = out58_
                        generated = d_81_wg6_
                        insideConstrainedOut = d_82_wi6_
                        currentConstrainedOut = d_83_wc6_
                        d_2_steps_ = (d_2_steps_) + (d_80_remainBudget6_)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_84_finalBudget_: int
            d_84_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_84_finalBudget_) > (0):
                d_85_wgf_: _dafny.Seq
                d_86_wif_: bool
                d_87_wcf_: _dafny.Seq
                out59_: _dafny.Seq
                out60_: bool
                out61_: _dafny.Seq
                out59_, out60_, out61_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_84_finalBudget_)
                d_85_wgf_ = out59_
                d_86_wif_ = out60_
                d_87_wcf_ = out61_
                generated = d_85_wgf_
                insideConstrainedOut = d_86_wif_
                currentConstrainedOut = d_87_wcf_
                d_2_steps_ = (d_2_steps_) + (d_84_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

