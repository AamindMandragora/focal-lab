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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step using the given variable names. At the very end, write your final answer as <<expression>> using plain variable names (no curly braces, write x not {x}), numbers, and operators +, -, *, /, //, %, int(). Example: <<n1 + n2>>, <<int(a * b / c)>>, <<(a + b) // c>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanBudget_: int
        d_3_spanBudget_ = 80
        d_4_nearBudgetThreshold_: int
        d_4_nearBudgetThreshold_ = 100
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingBudget_: int
                        d_5_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_5_remainingBudget_) <= (3):
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
                                d_10_remAfter_: int
                                d_10_remAfter_ = (maxSteps) - (d_1_steps_)
                                if ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and ((d_10_remAfter_) <= (d_4_nearBudgetThreshold_)):
                                    d_11_og2_: _dafny.Seq
                                    d_12_oi2_: bool
                                    d_13_oc2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_11_og2_ = out4_
                                    d_12_oi2_ = out5_
                                    d_13_oc2_ = out6_
                                    generated = d_11_og2_
                                    insideConstrainedOut = d_12_oi2_
                                    currentConstrainedOut = d_13_oc2_
                                    d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_14_remainingSteps_: int
                        d_14_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_14_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_15_closeBudget2_: int
                        if (d_14_remainingSteps_) < (30):
                            d_15_closeBudget2_ = d_14_remainingSteps_
                        elif True:
                            d_15_closeBudget2_ = 30
                        d_16_cg2_: _dafny.Seq
                        d_17_ci2_: bool
                        d_18_cc2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget2_)
                        d_16_cg2_ = out7_
                        d_17_ci2_ = out8_
                        d_18_cc2_ = out9_
                        generated = d_16_cg2_
                        insideConstrainedOut = d_17_ci2_
                        currentConstrainedOut = d_18_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_15_closeBudget2_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                        d_20_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_20_next_) == (eosToken):
                            d_21_remainingSteps_: int
                            d_21_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            if (d_21_remainingSteps_) == (0):
                                raise _dafny.Break("0")
                            d_22_closeBudget3_: int
                            if (d_21_remainingSteps_) < (20):
                                d_22_closeBudget3_ = d_21_remainingSteps_
                            elif True:
                                d_22_closeBudget3_ = 20
                            d_23_cg3_: _dafny.Seq
                            d_24_ci3_: bool
                            d_25_cc3_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget3_)
                            d_23_cg3_ = out11_
                            d_24_ci3_ = out12_
                            d_25_cc3_ = out13_
                            generated = d_23_cg3_
                            insideConstrainedOut = d_24_ci3_
                            currentConstrainedOut = d_25_cc3_
                            d_1_steps_ = (d_1_steps_) + (d_22_closeBudget3_)
                            raise _dafny.Break("0")
                        elif True:
                            d_26_isComplete_: bool
                            d_26_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_26_isComplete_:
                                d_27_remainingSteps_: int
                                d_27_remainingSteps_ = (maxSteps) - (d_1_steps_)
                                if (d_27_remainingSteps_) == (0):
                                    raise _dafny.Break("0")
                                d_28_closeBudget4_: int
                                if (d_27_remainingSteps_) < (20):
                                    d_28_closeBudget4_ = d_27_remainingSteps_
                                elif True:
                                    d_28_closeBudget4_ = 20
                                d_29_cg4_: _dafny.Seq
                                d_30_ci4_: bool
                                d_31_cc4_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_closeBudget4_)
                                d_29_cg4_ = out14_
                                d_30_ci4_ = out15_
                                d_31_cc4_ = out16_
                                generated = d_29_cg4_
                                insideConstrainedOut = d_30_ci4_
                                currentConstrainedOut = d_31_cc4_
                                d_1_steps_ = (d_1_steps_) + (d_28_closeBudget4_)
                                d_2_spanSteps_ = 0
                            elif True:
                                d_32_ag_: _dafny.Seq
                                d_33_ai_: bool
                                d_34_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                d_32_ag_ = out17_
                                d_33_ai_ = out18_
                                d_34_ac_ = out19_
                                generated = d_32_ag_
                                insideConstrainedOut = d_33_ai_
                                currentConstrainedOut = d_34_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_35_remainingA_: int
            d_35_remainingA_ = (maxSteps) - (d_1_steps_)
            d_36_closeBudgetA_: int
            if (d_35_remainingA_) < (50):
                d_36_closeBudgetA_ = d_35_remainingA_
            elif True:
                d_36_closeBudgetA_ = 50
            d_37_cgA_: _dafny.Seq
            d_38_ciA_: bool
            d_39_ccA_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: bool
            out22_: _dafny.Seq
            out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_36_closeBudgetA_)
            d_37_cgA_ = out20_
            d_38_ciA_ = out21_
            d_39_ccA_ = out22_
            generated = d_37_cgA_
            insideConstrainedOut = d_38_ciA_
            currentConstrainedOut = d_39_ccA_
            d_1_steps_ = (d_1_steps_) + (d_36_closeBudgetA_)
        d_40_genStr_: _dafny.Seq
        d_40_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_41_openCount_: int
        d_41_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_40_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_41_openCount_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_42_remainingB_: int
            d_42_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_42_remainingB_) >= (5):
                d_43_ogB_: _dafny.Seq
                d_44_oiB_: bool
                d_45_ocB_: _dafny.Seq
                out23_: _dafny.Seq
                out24_: bool
                out25_: _dafny.Seq
                out23_, out24_, out25_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_43_ogB_ = out23_
                d_44_oiB_ = out24_
                d_45_ocB_ = out25_
                generated = d_43_ogB_
                insideConstrainedOut = d_44_oiB_
                currentConstrainedOut = d_45_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_46_remainingB2_: int
                    d_46_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_47_closeBudgetB_: int
                    if (d_46_remainingB2_) < (90):
                        d_47_closeBudgetB_ = d_46_remainingB2_
                    elif True:
                        d_47_closeBudgetB_ = 90
                    if (d_47_closeBudgetB_) > (0):
                        d_48_cgB_: _dafny.Seq
                        d_49_ciB_: bool
                        d_50_ccB_: _dafny.Seq
                        out26_: _dafny.Seq
                        out27_: bool
                        out28_: _dafny.Seq
                        out26_, out27_, out28_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_47_closeBudgetB_)
                        d_48_cgB_ = out26_
                        d_49_ciB_ = out27_
                        d_50_ccB_ = out28_
                        generated = d_48_cgB_
                        insideConstrainedOut = d_49_ciB_
                        currentConstrainedOut = d_50_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_47_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

