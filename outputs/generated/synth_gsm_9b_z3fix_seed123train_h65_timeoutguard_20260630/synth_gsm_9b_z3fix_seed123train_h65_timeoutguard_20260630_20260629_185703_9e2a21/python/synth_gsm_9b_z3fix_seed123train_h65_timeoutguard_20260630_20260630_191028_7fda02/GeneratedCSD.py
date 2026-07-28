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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. For each intermediate calculation and the final answer, write the symbolic expression inside << >> delimiters. Use PLAIN variable names WITHOUT curly braces: write n not {n}, write price not {price}. Use only operators +, -, *, /, //, %, int(). No ** or ^ or {}. Example: <<n * price>>, <<int(a / b)>>, <<(x - y) * z // 60>>. The very last <<expression>> is extracted as your final answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_phase2Threshold_: int
        if (maxSteps) > (150):
            d_2_phase2Threshold_ = (maxSteps) - (150)
        elif True:
            d_2_phase2Threshold_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remainingBudget_: int
                        d_3_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_3_remainingBudget_) <= (2):
                            raise _dafny.Break("0")
                        elif (d_1_steps_) >= (d_2_phase2Threshold_):
                            d_4_og_: _dafny.Seq
                            d_5_oi_: bool
                            d_6_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_og_ = out0_
                            d_5_oi_ = out1_
                            d_6_oc_ = out2_
                            generated = d_4_og_
                            insideConstrainedOut = d_5_oi_
                            currentConstrainedOut = d_6_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                    elif True:
                        d_8_remainingBudget2_: int
                        d_8_remainingBudget2_ = (maxSteps) - (d_1_steps_)
                        if (d_8_remainingBudget2_) == (0):
                            raise _dafny.Break("0")
                        d_9_isComplete_: bool
                        d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_isComplete_:
                            d_10_cg4_: _dafny.Seq
                            d_11_ci4_: bool
                            d_12_cc4_: _dafny.Seq
                            d_13_closed4_: bool
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_10_cg4_ = out4_
                            d_11_ci4_ = out5_
                            d_12_cc4_ = out6_
                            d_13_closed4_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_13_closed4_:
                                generated = d_10_cg4_
                                insideConstrainedOut = d_11_ci4_
                                currentConstrainedOut = d_12_cc4_
                            elif True:
                                d_14_constrainedPrompt2_: _dafny.Seq
                                d_14_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_15_next2_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_14_constrainedPrompt2_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_15_next2_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_valid2_: bool
                                    out9_: bool
                                    out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_15_next2_)
                                    d_16_valid2_ = out9_
                                    if d_16_valid2_:
                                        d_17_ag2_: _dafny.Seq
                                        d_18_ai2_: bool
                                        d_19_ac2_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next2_)
                                        d_17_ag2_ = out10_
                                        d_18_ai2_ = out11_
                                        d_19_ac2_ = out12_
                                        generated = d_17_ag2_
                                        insideConstrainedOut = d_18_ai2_
                                        currentConstrainedOut = d_19_ac2_
                        elif True:
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_21_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_21_next_ = out13_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                d_22_remainingEos_: int
                                d_22_remainingEos_ = (maxSteps) - (d_1_steps_)
                                if (d_22_remainingEos_) > (0):
                                    d_23_closeBudgetEos_: int
                                    if (d_22_remainingEos_) < (30):
                                        d_23_closeBudgetEos_ = d_22_remainingEos_
                                    elif True:
                                        d_23_closeBudgetEos_ = 30
                                    d_24_cgE_: _dafny.Seq
                                    d_25_ciE_: bool
                                    d_26_ccE_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudgetEos_)
                                    d_24_cgE_ = out14_
                                    d_25_ciE_ = out15_
                                    d_26_ccE_ = out16_
                                    generated = d_24_cgE_
                                    insideConstrainedOut = d_25_ciE_
                                    currentConstrainedOut = d_26_ccE_
                                    d_1_steps_ = (d_1_steps_) + (d_23_closeBudgetEos_)
                                raise _dafny.Break("0")
                            elif True:
                                d_27_valid_: bool
                                out17_: bool
                                out17_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_21_next_)
                                d_27_valid_ = out17_
                                if d_27_valid_:
                                    d_28_ag_: _dafny.Seq
                                    d_29_ai_: bool
                                    d_30_ac_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_28_ag_ = out18_
                                    d_29_ai_ = out19_
                                    d_30_ac_ = out20_
                                    generated = d_28_ag_
                                    insideConstrainedOut = d_29_ai_
                                    currentConstrainedOut = d_30_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_31_remainingA_: int
            d_31_remainingA_ = (maxSteps) - (d_1_steps_)
            d_32_closeBudgetA_: int
            if (d_31_remainingA_) < (60):
                d_32_closeBudgetA_ = d_31_remainingA_
            elif True:
                d_32_closeBudgetA_ = 60
            if (d_32_closeBudgetA_) > (0):
                d_33_cgA_: _dafny.Seq
                d_34_ciA_: bool
                d_35_ccA_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudgetA_)
                d_33_cgA_ = out21_
                d_34_ciA_ = out22_
                d_35_ccA_ = out23_
                generated = d_33_cgA_
                insideConstrainedOut = d_34_ciA_
                currentConstrainedOut = d_35_ccA_
                d_1_steps_ = (d_1_steps_) + (d_32_closeBudgetA_)
        d_36_genStr_: _dafny.Seq
        d_36_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_37_openCount_: int
        d_37_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_36_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_37_openCount_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_38_remainingB_: int
            d_38_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_38_remainingB_) >= (3):
                d_39_ogB_: _dafny.Seq
                d_40_oiB_: bool
                d_41_ocB_: _dafny.Seq
                out24_: _dafny.Seq
                out25_: bool
                out26_: _dafny.Seq
                out24_, out25_, out26_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_39_ogB_ = out24_
                d_40_oiB_ = out25_
                d_41_ocB_ = out26_
                generated = d_39_ogB_
                insideConstrainedOut = d_40_oiB_
                currentConstrainedOut = d_41_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_42_remainingB2_: int
                    d_42_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_43_closeBudgetB_: int
                    if (d_42_remainingB2_) < (90):
                        d_43_closeBudgetB_ = d_42_remainingB2_
                    elif True:
                        d_43_closeBudgetB_ = 90
                    if (d_43_closeBudgetB_) > (0):
                        d_44_cgB_: _dafny.Seq
                        d_45_ciB_: bool
                        d_46_ccB_: _dafny.Seq
                        out27_: _dafny.Seq
                        out28_: bool
                        out29_: _dafny.Seq
                        out27_, out28_, out29_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_43_closeBudgetB_)
                        d_44_cgB_ = out27_
                        d_45_ciB_ = out28_
                        d_46_ccB_ = out29_
                        generated = d_44_cgB_
                        insideConstrainedOut = d_45_ciB_
                        currentConstrainedOut = d_46_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_43_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

