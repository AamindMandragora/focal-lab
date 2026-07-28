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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Variables like {cur}{p1} mean the price is p1 (not curp1). Use int() for final monetary answers. End with <<expression>> using plain variable names (e.g., n1, p1, k), numbers, +, -, *, /, //, %, int(). No {braces}, no **. Example: <<int(n * p)>>, <<(a + b) // c>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanBudget_: int
        d_3_spanBudget_ = 80
        d_4_nearBudgetThreshold_: int
        d_4_nearBudgetThreshold_ = 100
        d_5_earlyForceThreshold_: int
        d_5_earlyForceThreshold_ = 400
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_remainingBudget_: int
                        d_6_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_6_remainingBudget_) <= (3):
                            raise _dafny.Break("0")
                        d_7_genStr2_: _dafny.Seq
                        d_7_genStr2_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
                        d_8_openCount2_: int
                        d_8_openCount2_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_7_genStr2_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_9_shouldForce_: bool
                        d_9_shouldForce_ = (((d_1_steps_) >= (d_5_earlyForceThreshold_)) and ((d_8_openCount2_) == (0))) or (((d_6_remainingBudget_) <= (d_4_nearBudgetThreshold_)) and ((d_8_openCount2_) == (0)))
                        if d_9_shouldForce_:
                            d_10_og_: _dafny.Seq
                            d_11_oi_: bool
                            d_12_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_og_ = out0_
                            d_11_oi_ = out1_
                            d_12_oc_ = out2_
                            generated = d_10_og_
                            insideConstrainedOut = d_11_oi_
                            currentConstrainedOut = d_12_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = 0
                        elif True:
                            d_13_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_14_og2_: _dafny.Seq
                                    d_15_oi2_: bool
                                    d_16_oc2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_14_og2_ = out4_
                                    d_15_oi2_ = out5_
                                    d_16_oc2_ = out6_
                                    generated = d_14_og2_
                                    insideConstrainedOut = d_15_oi2_
                                    currentConstrainedOut = d_16_oc2_
                                    d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_17_remainingSteps_: int
                        d_17_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_17_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_18_closeBudget2_: int
                        if (d_17_remainingSteps_) < (30):
                            d_18_closeBudget2_ = d_17_remainingSteps_
                        elif True:
                            d_18_closeBudget2_ = 30
                        d_19_cg2_: _dafny.Seq
                        d_20_ci2_: bool
                        d_21_cc2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget2_)
                        d_19_cg2_ = out7_
                        d_20_ci2_ = out8_
                        d_21_cc2_ = out9_
                        generated = d_19_cg2_
                        insideConstrainedOut = d_20_ci2_
                        currentConstrainedOut = d_21_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_18_closeBudget2_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_22_cg1_: _dafny.Seq
                        d_23_ci1_: bool
                        d_24_cc1_: _dafny.Seq
                        d_25_closed1_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_22_cg1_ = out10_
                        d_23_ci1_ = out11_
                        d_24_cc1_ = out12_
                        d_25_closed1_ = out13_
                        if d_25_closed1_:
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_22_cg1_
                            insideConstrainedOut = d_23_ci1_
                            currentConstrainedOut = d_24_cc1_
                            d_2_spanSteps_ = 0
                        elif True:
                            d_26_constrainedPrompt_: _dafny.Seq
                            d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_27_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_27_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                            if (d_27_next_) == (eosToken):
                                d_28_remainingSteps_: int
                                d_28_remainingSteps_ = (maxSteps) - (d_1_steps_)
                                if (d_28_remainingSteps_) == (0):
                                    raise _dafny.Break("0")
                                d_29_closeBudget3_: int
                                if (d_28_remainingSteps_) < (25):
                                    d_29_closeBudget3_ = d_28_remainingSteps_
                                elif True:
                                    d_29_closeBudget3_ = 25
                                d_30_cg3_: _dafny.Seq
                                d_31_ci3_: bool
                                d_32_cc3_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget3_)
                                d_30_cg3_ = out15_
                                d_31_ci3_ = out16_
                                d_32_cc3_ = out17_
                                generated = d_30_cg3_
                                insideConstrainedOut = d_31_ci3_
                                currentConstrainedOut = d_32_cc3_
                                d_1_steps_ = (d_1_steps_) + (d_29_closeBudget3_)
                                d_2_spanSteps_ = 0
                                raise _dafny.Break("0")
                            elif True:
                                d_33_isComplete_: bool
                                d_33_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_33_isComplete_:
                                    d_34_remainingSteps_: int
                                    d_34_remainingSteps_ = (maxSteps) - (d_1_steps_)
                                    if (d_34_remainingSteps_) == (0):
                                        raise _dafny.Break("0")
                                    d_35_closeBudget4_: int
                                    if (d_34_remainingSteps_) < (20):
                                        d_35_closeBudget4_ = d_34_remainingSteps_
                                    elif True:
                                        d_35_closeBudget4_ = 20
                                    d_36_cg4_: _dafny.Seq
                                    d_37_ci4_: bool
                                    d_38_cc4_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_35_closeBudget4_)
                                    d_36_cg4_ = out18_
                                    d_37_ci4_ = out19_
                                    d_38_cc4_ = out20_
                                    generated = d_36_cg4_
                                    insideConstrainedOut = d_37_ci4_
                                    currentConstrainedOut = d_38_cc4_
                                    d_1_steps_ = (d_1_steps_) + (d_35_closeBudget4_)
                                    d_2_spanSteps_ = 0
                                elif True:
                                    d_39_ag_: _dafny.Seq
                                    d_40_ai_: bool
                                    d_41_ac_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                    d_39_ag_ = out21_
                                    d_40_ai_ = out22_
                                    d_41_ac_ = out23_
                                    generated = d_39_ag_
                                    insideConstrainedOut = d_40_ai_
                                    currentConstrainedOut = d_41_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_42_remainingA_: int
            d_42_remainingA_ = (maxSteps) - (d_1_steps_)
            d_43_closeBudgetA_: int
            if (d_42_remainingA_) < (60):
                d_43_closeBudgetA_ = d_42_remainingA_
            elif True:
                d_43_closeBudgetA_ = 60
            d_44_cgA_: _dafny.Seq
            d_45_ciA_: bool
            d_46_ccA_: _dafny.Seq
            out24_: _dafny.Seq
            out25_: bool
            out26_: _dafny.Seq
            out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_43_closeBudgetA_)
            d_44_cgA_ = out24_
            d_45_ciA_ = out25_
            d_46_ccA_ = out26_
            generated = d_44_cgA_
            insideConstrainedOut = d_45_ciA_
            currentConstrainedOut = d_46_ccA_
            d_1_steps_ = (d_1_steps_) + (d_43_closeBudgetA_)
        d_47_genStrFinal_: _dafny.Seq
        d_47_genStrFinal_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_48_openCountFinal_: int
        d_48_openCountFinal_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_47_genStrFinal_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_48_openCountFinal_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_49_remainingB_: int
            d_49_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_49_remainingB_) >= (5):
                d_50_ogB_: _dafny.Seq
                d_51_oiB_: bool
                d_52_ocB_: _dafny.Seq
                out27_: _dafny.Seq
                out28_: bool
                out29_: _dafny.Seq
                out27_, out28_, out29_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_50_ogB_ = out27_
                d_51_oiB_ = out28_
                d_52_ocB_ = out29_
                generated = d_50_ogB_
                insideConstrainedOut = d_51_oiB_
                currentConstrainedOut = d_52_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_53_remainingB2_: int
                    d_53_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_54_closeBudgetB_: int
                    if (d_53_remainingB2_) < (90):
                        d_54_closeBudgetB_ = d_53_remainingB2_
                    elif True:
                        d_54_closeBudgetB_ = 90
                    if (d_54_closeBudgetB_) > (0):
                        d_55_cgB_: _dafny.Seq
                        d_56_ciB_: bool
                        d_57_ccB_: _dafny.Seq
                        out30_: _dafny.Seq
                        out31_: bool
                        out32_: _dafny.Seq
                        out30_, out31_, out32_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_54_closeBudgetB_)
                        d_55_cgB_ = out30_
                        d_56_ciB_ = out31_
                        d_57_ccB_ = out32_
                        generated = d_55_cgB_
                        insideConstrainedOut = d_56_ciB_
                        currentConstrainedOut = d_57_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_54_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

