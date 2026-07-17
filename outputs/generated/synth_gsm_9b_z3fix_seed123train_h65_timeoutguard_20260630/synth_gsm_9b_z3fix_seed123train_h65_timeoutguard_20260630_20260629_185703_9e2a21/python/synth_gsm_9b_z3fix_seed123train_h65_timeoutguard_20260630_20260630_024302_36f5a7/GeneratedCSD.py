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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the given variable names.\n\nIMPORTANT: At the very end of your solution, write your FINAL ANSWER as <<expression>>.\n\nRules for the expression:\n- Use variable names exactly as given (no curly braces: write n1 not {n1})\n- AVAILABLE OPERATORS: + - * // % int() -- these are the ONLY operators\n- Use // for integer/floor division (e.g., total_minutes // 60 for whole hours)\n- Use int() when a multiplication by a fraction gives a whole-number count (e.g., int(n * frac))\n- Do NOT use / (use // or int() instead), do NOT use ** (not supported)\n- Write the COMPLETE single arithmetic expression for the final answer\n- The << >> span is for the final answer ONLY, not intermediate computations\n\nGood examples:\n  <<int(n1 * frac) + n2>>\n  <<(total_minutes) // 60>>\n  <<p1 + int(p2 * rate) - discount>>\n  <<n - n1 - 2 * n2>>\n  <<(w1 + w2 + w3) * price>>")))
        d_1_reserve_: int
        d_1_reserve_ = 120
        d_2_mainBudget_: int
        if (maxSteps) > (d_1_reserve_):
            d_2_mainBudget_ = (maxSteps) - (d_1_reserve_)
        elif True:
            d_2_mainBudget_ = 0
        d_3_steps_: int
        d_3_steps_ = 0
        d_4_spanSteps_: int
        d_4_spanSteps_ = 0
        d_5_spanBudget_: int
        d_5_spanBudget_ = 80
        d_6_nearBudgetThreshold_: int
        d_6_nearBudgetThreshold_ = 200
        with _dafny.label("0"):
            while (d_3_steps_) < (d_2_mainBudget_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_remainingTotal_: int
                        d_7_remainingTotal_ = (maxSteps) - (d_3_steps_)
                        if (d_7_remainingTotal_) <= (d_6_nearBudgetThreshold_):
                            d_8_og_: _dafny.Seq
                            d_9_oi_: bool
                            d_10_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_og_ = out0_
                            d_9_oi_ = out1_
                            d_10_oc_ = out2_
                            generated = d_8_og_
                            insideConstrainedOut = d_9_oi_
                            currentConstrainedOut = d_10_oc_
                            d_3_steps_ = (d_3_steps_) + (1)
                            d_4_spanSteps_ = 0
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_3_steps_ = (d_3_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                d_12_remAfter_: int
                                d_12_remAfter_ = (maxSteps) - (d_3_steps_)
                                if ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and ((d_12_remAfter_) <= (d_6_nearBudgetThreshold_)):
                                    d_13_og2_: _dafny.Seq
                                    d_14_oi2_: bool
                                    d_15_oc2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_13_og2_ = out4_
                                    d_14_oi2_ = out5_
                                    d_15_oc2_ = out6_
                                    generated = d_13_og2_
                                    insideConstrainedOut = d_14_oi2_
                                    currentConstrainedOut = d_15_oc2_
                                    d_4_spanSteps_ = 0
                    elif (d_4_spanSteps_) >= (d_5_spanBudget_):
                        d_16_remainingSteps_: int
                        d_16_remainingSteps_ = (d_2_mainBudget_) - (d_3_steps_)
                        if (d_16_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_17_closeBudget2_: int
                        if (d_16_remainingSteps_) < (25):
                            d_17_closeBudget2_ = d_16_remainingSteps_
                        elif True:
                            d_17_closeBudget2_ = 25
                        d_18_cg2_: _dafny.Seq
                        d_19_ci2_: bool
                        d_20_cc2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget2_)
                        d_18_cg2_ = out7_
                        d_19_ci2_ = out8_
                        d_20_cc2_ = out9_
                        generated = d_18_cg2_
                        insideConstrainedOut = d_19_ci2_
                        currentConstrainedOut = d_20_cc2_
                        d_3_steps_ = (d_3_steps_) + (d_17_closeBudget2_)
                        d_4_spanSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_21_remainingSteps_: int
                        d_21_remainingSteps_ = (d_2_mainBudget_) - (d_3_steps_)
                        if (d_21_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_22_closeBudget_: int
                        if (d_21_remainingSteps_) < (15):
                            d_22_closeBudget_ = d_21_remainingSteps_
                        elif True:
                            d_22_closeBudget_ = 15
                        d_23_cg_: _dafny.Seq
                        d_24_ci_: bool
                        d_25_cc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
                        d_23_cg_ = out10_
                        d_24_ci_ = out11_
                        d_25_cc_ = out12_
                        generated = d_23_cg_
                        insideConstrainedOut = d_24_ci_
                        currentConstrainedOut = d_25_cc_
                        d_3_steps_ = (d_3_steps_) + (d_22_closeBudget_)
                        d_4_spanSteps_ = 0
                    elif True:
                        d_26_constrainedPrompt_: _dafny.Seq
                        d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_27_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                        d_27_next_ = out13_
                        d_3_steps_ = (d_3_steps_) + (1)
                        d_4_spanSteps_ = (d_4_spanSteps_) + (1)
                        if (d_27_next_) == (eosToken):
                            d_28_remainingSteps_: int
                            d_28_remainingSteps_ = (d_2_mainBudget_) - (d_3_steps_)
                            if (d_28_remainingSteps_) == (0):
                                raise _dafny.Break("0")
                            d_29_closeBudget3_: int
                            if (d_28_remainingSteps_) < (20):
                                d_29_closeBudget3_ = d_28_remainingSteps_
                            elif True:
                                d_29_closeBudget3_ = 20
                            d_30_cg3_: _dafny.Seq
                            d_31_ci3_: bool
                            d_32_cc3_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget3_)
                            d_30_cg3_ = out14_
                            d_31_ci3_ = out15_
                            d_32_cc3_ = out16_
                            generated = d_30_cg3_
                            insideConstrainedOut = d_31_ci3_
                            currentConstrainedOut = d_32_cc3_
                            d_3_steps_ = (d_3_steps_) + (d_29_closeBudget3_)
                            raise _dafny.Break("0")
                        elif True:
                            d_33_ag_: _dafny.Seq
                            d_34_ai_: bool
                            d_35_ac_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                            d_33_ag_ = out17_
                            d_34_ai_ = out18_
                            d_35_ac_ = out19_
                            generated = d_33_ag_
                            insideConstrainedOut = d_34_ai_
                            currentConstrainedOut = d_35_ac_
                    pass
            pass
        if insideConstrainedOut:
            d_36_remainingA_: int
            d_36_remainingA_ = (maxSteps) - (d_3_steps_)
            d_37_closeBudgetA_: int
            if (d_36_remainingA_) < (50):
                d_37_closeBudgetA_ = d_36_remainingA_
            elif True:
                d_37_closeBudgetA_ = 50
            if (d_37_closeBudgetA_) > (0):
                d_38_cgA_: _dafny.Seq
                d_39_ciA_: bool
                d_40_ccA_: _dafny.Seq
                out20_: _dafny.Seq
                out21_: bool
                out22_: _dafny.Seq
                out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_37_closeBudgetA_)
                d_38_cgA_ = out20_
                d_39_ciA_ = out21_
                d_40_ccA_ = out22_
                generated = d_38_cgA_
                insideConstrainedOut = d_39_ciA_
                currentConstrainedOut = d_40_ccA_
                d_3_steps_ = (d_3_steps_) + (d_37_closeBudgetA_)
        d_41_genStr_: _dafny.Seq
        d_41_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_42_openCount_: int
        d_42_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_41_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if ((d_42_openCount_) == (0)) and (not(insideConstrainedOut)):
            d_43_remainingB_: int
            d_43_remainingB_ = (maxSteps) - (d_3_steps_)
            if (d_43_remainingB_) >= (10):
                d_44_ogB_: _dafny.Seq
                d_45_oiB_: bool
                d_46_ocB_: _dafny.Seq
                out23_: _dafny.Seq
                out24_: bool
                out25_: _dafny.Seq
                out23_, out24_, out25_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_44_ogB_ = out23_
                d_45_oiB_ = out24_
                d_46_ocB_ = out25_
                generated = d_44_ogB_
                insideConstrainedOut = d_45_oiB_
                currentConstrainedOut = d_46_ocB_
                d_3_steps_ = (d_3_steps_) + (1)
                d_47_remainingB2_: int
                d_47_remainingB2_ = (maxSteps) - (d_3_steps_)
                d_48_closeBudgetB_: int
                if (d_47_remainingB2_) < (69):
                    d_48_closeBudgetB_ = d_47_remainingB2_
                elif True:
                    d_48_closeBudgetB_ = 69
                if (d_48_closeBudgetB_) > (0):
                    d_49_cgB_: _dafny.Seq
                    d_50_ciB_: bool
                    d_51_ccB_: _dafny.Seq
                    out26_: _dafny.Seq
                    out27_: bool
                    out28_: _dafny.Seq
                    out26_, out27_, out28_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_48_closeBudgetB_)
                    d_49_cgB_ = out26_
                    d_50_ciB_ = out27_
                    d_51_ccB_ = out28_
                    generated = d_49_cgB_
                    insideConstrainedOut = d_50_ciB_
                    currentConstrainedOut = d_51_ccB_
                    d_3_steps_ = (d_3_steps_) + (d_48_closeBudgetB_)
        if (d_3_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

