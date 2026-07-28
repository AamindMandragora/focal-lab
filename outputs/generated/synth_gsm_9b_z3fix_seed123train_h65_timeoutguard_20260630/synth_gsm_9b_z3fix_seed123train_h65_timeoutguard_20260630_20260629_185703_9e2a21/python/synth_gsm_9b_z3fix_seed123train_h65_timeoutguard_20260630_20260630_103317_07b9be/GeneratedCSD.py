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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step using the given variable names (no curly braces: write x not {x}). Use ONLY plain variable names, numbers, +, -, *, /, //, %, int(), and parentheses in expressions. Do NOT use ** or LaTeX or {curly braces} in expressions. At the very end, write your final answer as <<expression>>. Examples: <<n1 + n2>>, <<int(a * b / c)>>, <<(n1 - n2) * l * p * t // 60>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanBudget_: int
        d_3_spanBudget_ = 60
        d_4_nearBudgetThreshold_: int
        d_4_nearBudgetThreshold_ = 120
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingBudget_: int
                        d_5_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_5_remainingBudget_) <= (2):
                            raise _dafny.Break("0")
                        elif (d_5_remainingBudget_) <= (d_4_nearBudgetThreshold_):
                            d_6_og_: _dafny.Seq
                            d_7_oi_: bool
                            d_8_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_og_ = out0_
                            d_7_oi_ = out1_
                            d_8_oc_ = out2_
                            generated = d_6_og_
                            insideConstrainedOut = d_7_oi_
                            currentConstrainedOut = d_8_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_10_og2_: _dafny.Seq
                                    d_11_oi2_: bool
                                    d_12_oc2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_10_og2_ = out4_
                                    d_11_oi2_ = out5_
                                    d_12_oc2_ = out6_
                                    generated = d_10_og2_
                                    insideConstrainedOut = d_11_oi2_
                                    currentConstrainedOut = d_12_oc2_
                                    d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_13_remainingSteps_: int
                        d_13_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_13_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_14_closeBudget2_: int
                        if (d_13_remainingSteps_) < (40):
                            d_14_closeBudget2_ = d_13_remainingSteps_
                        elif True:
                            d_14_closeBudget2_ = 40
                        d_15_cg2_: _dafny.Seq
                        d_16_ci2_: bool
                        d_17_cc2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget2_)
                        d_15_cg2_ = out7_
                        d_16_ci2_ = out8_
                        d_17_cc2_ = out9_
                        generated = d_15_cg2_
                        insideConstrainedOut = d_16_ci2_
                        currentConstrainedOut = d_17_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_14_closeBudget2_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_18_cg1_: _dafny.Seq
                        d_19_ci1_: bool
                        d_20_cc1_: _dafny.Seq
                        d_21_closed1_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_18_cg1_ = out10_
                        d_19_ci1_ = out11_
                        d_20_cc1_ = out12_
                        d_21_closed1_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if d_21_closed1_:
                            generated = d_18_cg1_
                            insideConstrainedOut = d_19_ci1_
                            currentConstrainedOut = d_20_cc1_
                            d_2_spanSteps_ = 0
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_23_next_ = out14_
                            if (d_23_next_) == (eosToken):
                                d_24_remainingSteps_: int
                                d_24_remainingSteps_ = (maxSteps) - (d_1_steps_)
                                if (d_24_remainingSteps_) == (0):
                                    raise _dafny.Break("0")
                                d_25_closeBudget3_: int
                                if (d_24_remainingSteps_) < (30):
                                    d_25_closeBudget3_ = d_24_remainingSteps_
                                elif True:
                                    d_25_closeBudget3_ = 30
                                d_26_cg3_: _dafny.Seq
                                d_27_ci3_: bool
                                d_28_cc3_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget3_)
                                d_26_cg3_ = out15_
                                d_27_ci3_ = out16_
                                d_28_cc3_ = out17_
                                generated = d_26_cg3_
                                insideConstrainedOut = d_27_ci3_
                                currentConstrainedOut = d_28_cc3_
                                d_1_steps_ = (d_1_steps_) + (d_25_closeBudget3_)
                                d_2_spanSteps_ = 0
                                raise _dafny.Break("0")
                            elif True:
                                d_29_ag_: _dafny.Seq
                                d_30_ai_: bool
                                d_31_ac_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_29_ag_ = out18_
                                d_30_ai_ = out19_
                                d_31_ac_ = out20_
                                generated = d_29_ag_
                                insideConstrainedOut = d_30_ai_
                                currentConstrainedOut = d_31_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_32_remainingA_: int
            d_32_remainingA_ = (maxSteps) - (d_1_steps_)
            d_33_closeBudgetA_: int
            if (d_32_remainingA_) < (60):
                d_33_closeBudgetA_ = d_32_remainingA_
            elif True:
                d_33_closeBudgetA_ = 60
            d_34_cgA_: _dafny.Seq
            d_35_ciA_: bool
            d_36_ccA_: _dafny.Seq
            out21_: _dafny.Seq
            out22_: bool
            out23_: _dafny.Seq
            out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudgetA_)
            d_34_cgA_ = out21_
            d_35_ciA_ = out22_
            d_36_ccA_ = out23_
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
                out24_: _dafny.Seq
                out25_: bool
                out26_: _dafny.Seq
                out24_, out25_, out26_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_40_ogB_ = out24_
                d_41_oiB_ = out25_
                d_42_ocB_ = out26_
                generated = d_40_ogB_
                insideConstrainedOut = d_41_oiB_
                currentConstrainedOut = d_42_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_43_remainingB2_: int
                    d_43_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_44_closeBudgetB_: int
                    if (d_43_remainingB2_) < (100):
                        d_44_closeBudgetB_ = d_43_remainingB2_
                    elif True:
                        d_44_closeBudgetB_ = 100
                    if (d_44_closeBudgetB_) > (0):
                        d_45_cgB_: _dafny.Seq
                        d_46_ciB_: bool
                        d_47_ccB_: _dafny.Seq
                        out27_: _dafny.Seq
                        out28_: bool
                        out29_: _dafny.Seq
                        out27_, out28_, out29_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_44_closeBudgetB_)
                        d_45_cgB_ = out27_
                        d_46_ciB_ = out28_
                        d_47_ccB_ = out29_
                        generated = d_45_cgB_
                        insideConstrainedOut = d_46_ciB_
                        currentConstrainedOut = d_47_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_44_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

