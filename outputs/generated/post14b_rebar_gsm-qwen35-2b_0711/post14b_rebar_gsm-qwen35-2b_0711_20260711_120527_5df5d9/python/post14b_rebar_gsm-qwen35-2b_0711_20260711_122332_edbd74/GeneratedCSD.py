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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step, using the exact variable names from the problem (like n, price, frac_1, etc.). At the very end, write: The answer is <<EXPR>> where EXPR is a single arithmetic expression combining variables with +, -, *, /. Do NOT use floor(), int(), ^, **, LaTeX, braces {}, or dollar signs. Just plain arithmetic with the variable names. Example: The answer is <<n * frac_1 * frac_2>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_hasCompletedSpan_: bool
        d_3_hasCompletedSpan_ = False
        d_4_phase1Limit_: int
        d_4_phase1Limit_ = _dafny.euclidian_division((maxSteps) * (80), 100)
        if ((d_4_phase1Limit_) < (20)) and ((maxSteps) >= (20)):
            d_4_phase1Limit_ = 20
        if (d_4_phase1Limit_) > (maxSteps):
            d_4_phase1Limit_ = maxSteps
        d_5_chunkSize_: int
        d_5_chunkSize_ = 50
        with _dafny.label("0"):
            while (((d_2_steps_) < (d_4_phase1Limit_)) and (not(insideConstrainedOut))) and (not(d_3_hasCompletedSpan_)):
                with _dafny.c_label("0"):
                    d_6_actualChunk_: int
                    d_6_actualChunk_ = d_5_chunkSize_
                    if ((d_2_steps_) + (d_6_actualChunk_)) > (d_4_phase1Limit_):
                        d_6_actualChunk_ = (d_4_phase1Limit_) - (d_2_steps_)
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
        d_14_minSpanTokens_: int
        d_14_minSpanTokens_ = 3
        d_15_spanTokensGenerated_: int
        d_15_spanTokensGenerated_ = 0
        d_16_innerStepLimit_: int
        d_16_innerStepLimit_ = 50
        d_17_innerSteps_: int
        d_17_innerSteps_ = 0
        with _dafny.label("1"):
            while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_17_innerSteps_) < (d_16_innerStepLimit_)):
                with _dafny.c_label("1"):
                    if ((d_15_spanTokensGenerated_) >= (d_14_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
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
                            out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_23_next_ = out11_
                            d_17_innerSteps_ = (d_17_innerSteps_) + (1)
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
                                d_15_spanTokensGenerated_ = (d_15_spanTokensGenerated_) + (1)
                    elif True:
                        d_27_constrainedPrompt_: _dafny.Seq
                        d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_28_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
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
                            d_15_spanTokensGenerated_ = (d_15_spanTokensGenerated_) + (1)
                    pass
            pass
        if ((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            d_32_spanBudget_: int
            d_32_spanBudget_ = 30
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
        if ((not(insideConstrainedOut)) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps)):
            if ((d_2_steps_) + (2)) <= (maxSteps):
                d_37_fg_: _dafny.Seq
                d_38_fi_: bool
                d_39_fc_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: bool
                out24_: _dafny.Seq
                out22_, out23_, out24_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_37_fg_ = out22_
                d_38_fi_ = out23_
                d_39_fc_ = out24_
                generated = d_37_fg_
                insideConstrainedOut = d_38_fi_
                currentConstrainedOut = d_39_fc_
                d_2_steps_ = (d_2_steps_) + (1)
                d_15_spanTokensGenerated_ = 0
                d_40_innerSteps4_: int
                d_40_innerSteps4_ = 0
                d_41_innerLimit4_: int
                d_41_innerLimit4_ = 40
                with _dafny.label("5_0_0"):
                    while (((insideConstrainedOut) and (not(d_3_hasCompletedSpan_))) and ((d_2_steps_) < (maxSteps))) and ((d_40_innerSteps4_) < (d_41_innerLimit4_)):
                        with _dafny.c_label("5_0_0"):
                            if ((d_15_spanTokensGenerated_) >= (d_14_minSpanTokens_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_42_cg4_: _dafny.Seq
                                d_43_ci4_: bool
                                d_44_cc4_: _dafny.Seq
                                d_45_closed4_: bool
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out28_: bool
                                out25_, out26_, out27_, out28_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_42_cg4_ = out25_
                                d_43_ci4_ = out26_
                                d_44_cc4_ = out27_
                                d_45_closed4_ = out28_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if d_45_closed4_:
                                    generated = d_42_cg4_
                                    insideConstrainedOut = d_43_ci4_
                                    currentConstrainedOut = d_44_cc4_
                                    d_3_hasCompletedSpan_ = True
                                elif True:
                                    d_46_constrainedPrompt4_: _dafny.Seq
                                    d_46_constrainedPrompt4_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_47_next4_: _dafny.Seq
                                    out29_: _dafny.Seq
                                    out29_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_46_constrainedPrompt4_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                    d_47_next4_ = out29_
                                    d_40_innerSteps4_ = (d_40_innerSteps4_) + (1)
                                    if (d_47_next4_) == (eosToken):
                                        raise _dafny.Break("5_0_0")
                                    elif True:
                                        d_48_ag4_: _dafny.Seq
                                        d_49_ai4_: bool
                                        d_50_ac4_: _dafny.Seq
                                        out30_: _dafny.Seq
                                        out31_: bool
                                        out32_: _dafny.Seq
                                        out30_, out31_, out32_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_47_next4_)
                                        d_48_ag4_ = out30_
                                        d_49_ai4_ = out31_
                                        d_50_ac4_ = out32_
                                        generated = d_48_ag4_
                                        insideConstrainedOut = d_49_ai4_
                                        currentConstrainedOut = d_50_ac4_
                                        d_15_spanTokensGenerated_ = (d_15_spanTokensGenerated_) + (1)
                            elif True:
                                d_51_constrainedPrompt4_: _dafny.Seq
                                d_51_constrainedPrompt4_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_52_next4_: _dafny.Seq
                                out33_: _dafny.Seq
                                out33_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_51_constrainedPrompt4_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_52_next4_ = out33_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_40_innerSteps4_ = (d_40_innerSteps4_) + (1)
                                if (d_52_next4_) == (eosToken):
                                    raise _dafny.Break("5_0_0")
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
                                    d_15_spanTokensGenerated_ = (d_15_spanTokensGenerated_) + (1)
                            pass
                    pass
                if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
                    d_56_remainBudget_: int
                    d_56_remainBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_56_remainBudget_) > (30):
                        d_56_remainBudget_ = 30
                    if (d_56_remainBudget_) > (0):
                        d_57_wg4_: _dafny.Seq
                        d_58_wi4_: bool
                        d_59_wc4_: _dafny.Seq
                        out37_: _dafny.Seq
                        out38_: bool
                        out39_: _dafny.Seq
                        out37_, out38_, out39_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_56_remainBudget_)
                        d_57_wg4_ = out37_
                        d_58_wi4_ = out38_
                        d_59_wc4_ = out39_
                        generated = d_57_wg4_
                        insideConstrainedOut = d_58_wi4_
                        currentConstrainedOut = d_59_wc4_
                        d_2_steps_ = (d_2_steps_) + (d_56_remainBudget_)
                        if not(insideConstrainedOut):
                            d_3_hasCompletedSpan_ = True
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_60_finalBudget_: int
            d_60_finalBudget_ = (maxSteps) - (d_2_steps_)
            if (d_60_finalBudget_) > (0):
                d_61_wgf_: _dafny.Seq
                d_62_wif_: bool
                d_63_wcf_: _dafny.Seq
                out40_: _dafny.Seq
                out41_: bool
                out42_: _dafny.Seq
                out40_, out41_, out42_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_60_finalBudget_)
                d_61_wgf_ = out40_
                d_62_wif_ = out41_
                d_63_wcf_ = out42_
                generated = d_61_wgf_
                insideConstrainedOut = d_62_wif_
                currentConstrainedOut = d_63_wcf_
                d_2_steps_ = (d_2_steps_) + (d_60_finalBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

