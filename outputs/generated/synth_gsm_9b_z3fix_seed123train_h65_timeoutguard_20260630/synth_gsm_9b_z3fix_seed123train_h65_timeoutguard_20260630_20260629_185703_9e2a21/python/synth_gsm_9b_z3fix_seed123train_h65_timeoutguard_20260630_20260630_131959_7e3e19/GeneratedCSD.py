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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write exactly ONE final answer as <<expression>> using plain variable names (no curly braces), numbers, and operators +, -, *, /, //, %, int(). Do not use ** or ^. Example: <<n * p>>, <<int(a + b)>>, <<(x + y) // z>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanBudget_: int
        d_3_spanBudget_ = 70
        d_4_forceOpenThreshold_: int
        d_4_forceOpenThreshold_ = 80
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingBudget_: int
                        d_5_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_5_remainingBudget_) <= (3):
                            raise _dafny.Break("0")
                        elif (d_5_remainingBudget_) <= (d_4_forceOpenThreshold_):
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
                                    d_13_rem_: int
                                    d_13_rem_ = (maxSteps) - (d_1_steps_)
                                    if (d_13_rem_) > (0):
                                        d_14_cb_: int
                                        if (d_13_rem_) < (d_3_spanBudget_):
                                            d_14_cb_ = d_13_rem_
                                        elif True:
                                            d_14_cb_ = d_3_spanBudget_
                                        d_15_cg_: _dafny.Seq
                                        d_16_ci_: bool
                                        d_17_cc_: _dafny.Seq
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out9_: _dafny.Seq
                                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_cb_)
                                        d_15_cg_ = out7_
                                        d_16_ci_ = out8_
                                        d_17_cc_ = out9_
                                        generated = d_15_cg_
                                        insideConstrainedOut = d_16_ci_
                                        currentConstrainedOut = d_17_cc_
                                        d_1_steps_ = (d_1_steps_) + (d_14_cb_)
                                        d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_18_remainingSteps_: int
                        d_18_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_18_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_19_closeBudget2_: int
                        if (d_18_remainingSteps_) < (30):
                            d_19_closeBudget2_ = d_18_remainingSteps_
                        elif True:
                            d_19_closeBudget2_ = 30
                        d_20_cg2_: _dafny.Seq
                        d_21_ci2_: bool
                        d_22_cc2_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget2_)
                        d_20_cg2_ = out10_
                        d_21_ci2_ = out11_
                        d_22_cc2_ = out12_
                        generated = d_20_cg2_
                        insideConstrainedOut = d_21_ci2_
                        currentConstrainedOut = d_22_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_19_closeBudget2_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_24_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                        d_24_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_24_next_) == (eosToken):
                            d_25_remainingSteps_: int
                            d_25_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            if (d_25_remainingSteps_) == (0):
                                raise _dafny.Break("0")
                            d_26_closeBudget3_: int
                            if (d_25_remainingSteps_) < (20):
                                d_26_closeBudget3_ = d_25_remainingSteps_
                            elif True:
                                d_26_closeBudget3_ = 20
                            d_27_cg3_: _dafny.Seq
                            d_28_ci3_: bool
                            d_29_cc3_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget3_)
                            d_27_cg3_ = out14_
                            d_28_ci3_ = out15_
                            d_29_cc3_ = out16_
                            generated = d_27_cg3_
                            insideConstrainedOut = d_28_ci3_
                            currentConstrainedOut = d_29_cc3_
                            d_1_steps_ = (d_1_steps_) + (d_26_closeBudget3_)
                            raise _dafny.Break("0")
                        elif True:
                            d_30_isComplete_: bool
                            d_30_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_30_isComplete_:
                                d_31_remainingSteps_: int
                                d_31_remainingSteps_ = (maxSteps) - (d_1_steps_)
                                if (d_31_remainingSteps_) == (0):
                                    raise _dafny.Break("0")
                                d_32_closeBudget4_: int
                                if (d_31_remainingSteps_) < (20):
                                    d_32_closeBudget4_ = d_31_remainingSteps_
                                elif True:
                                    d_32_closeBudget4_ = 20
                                d_33_cg4_: _dafny.Seq
                                d_34_ci4_: bool
                                d_35_cc4_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudget4_)
                                d_33_cg4_ = out17_
                                d_34_ci4_ = out18_
                                d_35_cc4_ = out19_
                                generated = d_33_cg4_
                                insideConstrainedOut = d_34_ci4_
                                currentConstrainedOut = d_35_cc4_
                                d_1_steps_ = (d_1_steps_) + (d_32_closeBudget4_)
                                d_2_spanSteps_ = 0
                            elif True:
                                d_36_ag_: _dafny.Seq
                                d_37_ai_: bool
                                d_38_ac_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_36_ag_ = out20_
                                d_37_ai_ = out21_
                                d_38_ac_ = out22_
                                generated = d_36_ag_
                                insideConstrainedOut = d_37_ai_
                                currentConstrainedOut = d_38_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_39_remainingA_: int
            d_39_remainingA_ = (maxSteps) - (d_1_steps_)
            d_40_closeBudgetA_: int
            if (d_39_remainingA_) < (50):
                d_40_closeBudgetA_ = d_39_remainingA_
            elif True:
                d_40_closeBudgetA_ = 50
            d_41_cgA_: _dafny.Seq
            d_42_ciA_: bool
            d_43_ccA_: _dafny.Seq
            out23_: _dafny.Seq
            out24_: bool
            out25_: _dafny.Seq
            out23_, out24_, out25_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_40_closeBudgetA_)
            d_41_cgA_ = out23_
            d_42_ciA_ = out24_
            d_43_ccA_ = out25_
            generated = d_41_cgA_
            insideConstrainedOut = d_42_ciA_
            currentConstrainedOut = d_43_ccA_
            d_1_steps_ = (d_1_steps_) + (d_40_closeBudgetA_)
        d_44_genStr_: _dafny.Seq
        d_44_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_45_openCount_: int
        d_45_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_44_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_45_openCount_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_46_remainingB_: int
            d_46_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_46_remainingB_) >= (5):
                d_47_ogB_: _dafny.Seq
                d_48_oiB_: bool
                d_49_ocB_: _dafny.Seq
                out26_: _dafny.Seq
                out27_: bool
                out28_: _dafny.Seq
                out26_, out27_, out28_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_47_ogB_ = out26_
                d_48_oiB_ = out27_
                d_49_ocB_ = out28_
                generated = d_47_ogB_
                insideConstrainedOut = d_48_oiB_
                currentConstrainedOut = d_49_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_50_remainingB2_: int
                    d_50_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_51_closeBudgetB_: int
                    if (d_50_remainingB2_) < (80):
                        d_51_closeBudgetB_ = d_50_remainingB2_
                    elif True:
                        d_51_closeBudgetB_ = 80
                    if (d_51_closeBudgetB_) > (0):
                        d_52_cgB_: _dafny.Seq
                        d_53_ciB_: bool
                        d_54_ccB_: _dafny.Seq
                        out29_: _dafny.Seq
                        out30_: bool
                        out31_: _dafny.Seq
                        out29_, out30_, out31_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_51_closeBudgetB_)
                        d_52_cgB_ = out29_
                        d_53_ciB_ = out30_
                        d_54_ccB_ = out31_
                        generated = d_52_cgB_
                        insideConstrainedOut = d_53_ciB_
                        currentConstrainedOut = d_54_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_51_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

