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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step using the variable names given in the problem. Write your final answer as <<expression>> using plain variable names (no curly braces: write n not {n}), numbers, and operators +, -, *, /, //, %, int(). Only the final answer goes inside << >>. Do not put reasoning inside << >>. Examples: <<n1 + n2>>, <<int(a * b / c)>>, <<count * (n1 + n2 + n3)>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanBudget_: int
        d_3_spanBudget_ = 60
        d_4_lateInterceptThreshold_: int
        d_4_lateInterceptThreshold_ = 80
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingBudget_: int
                        d_5_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_5_remainingBudget_) <= (3):
                            raise _dafny.Break("0")
                        d_6_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            d_7_remAfter_: int
                            d_7_remAfter_ = (maxSteps) - (d_1_steps_)
                            if ((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and ((d_7_remAfter_) <= (d_4_lateInterceptThreshold_)):
                                d_8_og2_: _dafny.Seq
                                d_9_oi2_: bool
                                d_10_oc2_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_8_og2_ = out1_
                                d_9_oi2_ = out2_
                                d_10_oc2_ = out3_
                                generated = d_8_og2_
                                insideConstrainedOut = d_9_oi2_
                                currentConstrainedOut = d_10_oc2_
                                d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_11_remainingSteps_: int
                        d_11_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_11_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_12_closeBudget2_: int
                        if (d_11_remainingSteps_) < (30):
                            d_12_closeBudget2_ = d_11_remainingSteps_
                        elif True:
                            d_12_closeBudget2_ = 30
                        d_13_cg2_: _dafny.Seq
                        d_14_ci2_: bool
                        d_15_cc2_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget2_)
                        d_13_cg2_ = out4_
                        d_14_ci2_ = out5_
                        d_15_cc2_ = out6_
                        generated = d_13_cg2_
                        insideConstrainedOut = d_14_ci2_
                        currentConstrainedOut = d_15_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_12_closeBudget2_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_17_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_17_next_) == (eosToken):
                            d_18_remainingSteps_: int
                            d_18_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            if (d_18_remainingSteps_) == (0):
                                raise _dafny.Break("0")
                            d_19_closeBudget3_: int
                            if (d_18_remainingSteps_) < (25):
                                d_19_closeBudget3_ = d_18_remainingSteps_
                            elif True:
                                d_19_closeBudget3_ = 25
                            d_20_cg3_: _dafny.Seq
                            d_21_ci3_: bool
                            d_22_cc3_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget3_)
                            d_20_cg3_ = out8_
                            d_21_ci3_ = out9_
                            d_22_cc3_ = out10_
                            generated = d_20_cg3_
                            insideConstrainedOut = d_21_ci3_
                            currentConstrainedOut = d_22_cc3_
                            d_1_steps_ = (d_1_steps_) + (d_19_closeBudget3_)
                            raise _dafny.Break("0")
                        elif True:
                            d_23_isComplete_: bool
                            d_23_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_23_isComplete_:
                                d_24_remainingSteps_: int
                                d_24_remainingSteps_ = (maxSteps) - (d_1_steps_)
                                if (d_24_remainingSteps_) == (0):
                                    raise _dafny.Break("0")
                                d_25_closeBudget4_: int
                                if (d_24_remainingSteps_) < (20):
                                    d_25_closeBudget4_ = d_24_remainingSteps_
                                elif True:
                                    d_25_closeBudget4_ = 20
                                d_26_cg4_: _dafny.Seq
                                d_27_ci4_: bool
                                d_28_cc4_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget4_)
                                d_26_cg4_ = out11_
                                d_27_ci4_ = out12_
                                d_28_cc4_ = out13_
                                generated = d_26_cg4_
                                insideConstrainedOut = d_27_ci4_
                                currentConstrainedOut = d_28_cc4_
                                d_1_steps_ = (d_1_steps_) + (d_25_closeBudget4_)
                                d_2_spanSteps_ = 0
                            elif True:
                                d_29_ag_: _dafny.Seq
                                d_30_ai_: bool
                                d_31_ac_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_29_ag_ = out14_
                                d_30_ai_ = out15_
                                d_31_ac_ = out16_
                                generated = d_29_ag_
                                insideConstrainedOut = d_30_ai_
                                currentConstrainedOut = d_31_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_32_remainingA_: int
            d_32_remainingA_ = (maxSteps) - (d_1_steps_)
            d_33_closeBudgetA_: int
            if (d_32_remainingA_) < (50):
                d_33_closeBudgetA_ = d_32_remainingA_
            elif True:
                d_33_closeBudgetA_ = 50
            d_34_cgA_: _dafny.Seq
            d_35_ciA_: bool
            d_36_ccA_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: bool
            out19_: _dafny.Seq
            out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudgetA_)
            d_34_cgA_ = out17_
            d_35_ciA_ = out18_
            d_36_ccA_ = out19_
            generated = d_34_cgA_
            insideConstrainedOut = d_35_ciA_
            currentConstrainedOut = d_36_ccA_
            d_1_steps_ = (d_1_steps_) + (d_33_closeBudgetA_)
        d_37_genStr_: _dafny.Seq
        d_37_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_38_openCount_: int
        d_38_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_37_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_38_openCount_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_39_remainingB_: int
            d_39_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_39_remainingB_) >= (5):
                d_40_ogB_: _dafny.Seq
                d_41_oiB_: bool
                d_42_ocB_: _dafny.Seq
                out20_: _dafny.Seq
                out21_: bool
                out22_: _dafny.Seq
                out20_, out21_, out22_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_40_ogB_ = out20_
                d_41_oiB_ = out21_
                d_42_ocB_ = out22_
                generated = d_40_ogB_
                insideConstrainedOut = d_41_oiB_
                currentConstrainedOut = d_42_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_43_remainingB2_: int
                    d_43_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_44_closeBudgetB_: int
                    if (d_43_remainingB2_) < (80):
                        d_44_closeBudgetB_ = d_43_remainingB2_
                    elif True:
                        d_44_closeBudgetB_ = 80
                    if (d_44_closeBudgetB_) > (0):
                        d_45_cgB_: _dafny.Seq
                        d_46_ciB_: bool
                        d_47_ccB_: _dafny.Seq
                        out23_: _dafny.Seq
                        out24_: bool
                        out25_: _dafny.Seq
                        out23_, out24_, out25_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_44_closeBudgetB_)
                        d_45_cgB_ = out23_
                        d_46_ciB_ = out24_
                        d_47_ccB_ = out25_
                        generated = d_45_cgB_
                        insideConstrainedOut = d_46_ciB_
                        currentConstrainedOut = d_47_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_44_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

