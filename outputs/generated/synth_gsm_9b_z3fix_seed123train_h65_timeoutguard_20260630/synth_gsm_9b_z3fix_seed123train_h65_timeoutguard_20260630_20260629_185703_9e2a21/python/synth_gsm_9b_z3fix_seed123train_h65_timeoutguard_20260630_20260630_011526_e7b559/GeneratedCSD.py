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
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
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
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_remainingSteps_: int
                        d_16_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_16_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_17_closeBudget_: int
                        if (d_16_remainingSteps_) < (15):
                            d_17_closeBudget_ = d_16_remainingSteps_
                        elif True:
                            d_17_closeBudget_ = 15
                        d_18_cg_: _dafny.Seq
                        d_19_ci_: bool
                        d_20_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                        d_18_cg_ = out7_
                        d_19_ci_ = out8_
                        d_20_cc_ = out9_
                        generated = d_18_cg_
                        insideConstrainedOut = d_19_ci_
                        currentConstrainedOut = d_20_cc_
                        d_1_steps_ = (d_1_steps_) + (d_17_closeBudget_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                        d_22_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_22_next_) == (eosToken):
                            d_23_remainingSteps_: int
                            d_23_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            if (d_23_remainingSteps_) == (0):
                                raise _dafny.Break("0")
                            d_24_closeBudget3_: int
                            if (d_23_remainingSteps_) < (20):
                                d_24_closeBudget3_ = d_23_remainingSteps_
                            elif True:
                                d_24_closeBudget3_ = 20
                            d_25_cg3_: _dafny.Seq
                            d_26_ci3_: bool
                            d_27_cc3_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget3_)
                            d_25_cg3_ = out11_
                            d_26_ci3_ = out12_
                            d_27_cc3_ = out13_
                            generated = d_25_cg3_
                            insideConstrainedOut = d_26_ci3_
                            currentConstrainedOut = d_27_cc3_
                            d_1_steps_ = (d_1_steps_) + (d_24_closeBudget3_)
                            raise _dafny.Break("0")
                        elif True:
                            d_28_ag_: _dafny.Seq
                            d_29_ai_: bool
                            d_30_ac_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                            d_28_ag_ = out14_
                            d_29_ai_ = out15_
                            d_30_ac_ = out16_
                            generated = d_28_ag_
                            insideConstrainedOut = d_29_ai_
                            currentConstrainedOut = d_30_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_31_remainingA_: int
            d_31_remainingA_ = (maxSteps) - (d_1_steps_)
            d_32_closeBudgetA_: int
            if (d_31_remainingA_) < (50):
                d_32_closeBudgetA_ = d_31_remainingA_
            elif True:
                d_32_closeBudgetA_ = 50
            d_33_cgA_: _dafny.Seq
            d_34_ciA_: bool
            d_35_ccA_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: bool
            out19_: _dafny.Seq
            out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudgetA_)
            d_33_cgA_ = out17_
            d_34_ciA_ = out18_
            d_35_ccA_ = out19_
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
            if (d_38_remainingB_) >= (5):
                d_39_ogB_: _dafny.Seq
                d_40_oiB_: bool
                d_41_ocB_: _dafny.Seq
                out20_: _dafny.Seq
                out21_: bool
                out22_: _dafny.Seq
                out20_, out21_, out22_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_39_ogB_ = out20_
                d_40_oiB_ = out21_
                d_41_ocB_ = out22_
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
                        out23_: _dafny.Seq
                        out24_: bool
                        out25_: _dafny.Seq
                        out23_, out24_, out25_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_43_closeBudgetB_)
                        d_44_cgB_ = out23_
                        d_45_ciB_ = out24_
                        d_46_ccB_ = out25_
                        generated = d_44_cgB_
                        insideConstrainedOut = d_45_ciB_
                        currentConstrainedOut = d_46_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_43_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

