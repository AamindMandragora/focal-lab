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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Write intermediate symbolic expressions and the final answer inside <<expression>> delimiters. Use plain variable names from the problem (no curly braces). Use operators: +, -, *, /, //, %. Use int() for integer division results. Do NOT use ** or ^. Write a single expression per span. Example: <<n * price>>, <<int((a + b) // c)>>, <<(x - y) * z>>. The last <<expression>> is the final answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanBudget_: int
        d_2_spanBudget_ = 50
        d_3_forceOpenThreshold_: int
        d_3_forceOpenThreshold_ = 85
        d_4_lastTokenWasLt_: bool
        d_4_lastTokenWasLt_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingBudget_: int
                        d_5_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_5_remainingBudget_) <= (3):
                            raise _dafny.Break("0")
                        elif (d_5_remainingBudget_) <= (d_3_forceOpenThreshold_):
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
                            d_4_lastTokenWasLt_ = False
                            if (d_1_steps_) < (maxSteps):
                                d_9_rem_: int
                                d_9_rem_ = (maxSteps) - (d_1_steps_)
                                d_10_cb_: int
                                if (d_9_rem_) < (d_2_spanBudget_):
                                    d_10_cb_ = d_9_rem_
                                elif True:
                                    d_10_cb_ = d_2_spanBudget_
                                if (d_10_cb_) > (0):
                                    d_11_cg_: _dafny.Seq
                                    d_12_ci_: bool
                                    d_13_cc_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out4_: bool
                                    out5_: _dafny.Seq
                                    out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_cb_)
                                    d_11_cg_ = out3_
                                    d_12_ci_ = out4_
                                    d_13_cc_ = out5_
                                    generated = d_11_cg_
                                    insideConstrainedOut = d_12_ci_
                                    currentConstrainedOut = d_13_cc_
                                    d_1_steps_ = (d_1_steps_) + (d_10_cb_)
                        elif True:
                            d_14_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                                d_4_lastTokenWasLt_ = False
                                d_15_og2_: _dafny.Seq
                                d_16_oi2_: bool
                                d_17_oc2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_15_og2_ = out7_
                                d_16_oi2_ = out8_
                                d_17_oc2_ = out9_
                                generated = d_15_og2_
                                insideConstrainedOut = d_16_oi2_
                                currentConstrainedOut = d_17_oc2_
                                d_18_rem2_: int
                                d_18_rem2_ = (maxSteps) - (d_1_steps_)
                                if (d_18_rem2_) > (0):
                                    d_19_cb2_: int
                                    if (d_18_rem2_) < (d_2_spanBudget_):
                                        d_19_cb2_ = d_18_rem2_
                                    elif True:
                                        d_19_cb2_ = d_2_spanBudget_
                                    d_20_cg2_: _dafny.Seq
                                    d_21_ci2_: bool
                                    d_22_cc2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_cb2_)
                                    d_20_cg2_ = out10_
                                    d_21_ci2_ = out11_
                                    d_22_cc2_ = out12_
                                    generated = d_20_cg2_
                                    insideConstrainedOut = d_21_ci2_
                                    currentConstrainedOut = d_22_cc2_
                                    d_1_steps_ = (d_1_steps_) + (d_19_cb2_)
                            elif ((d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")))) and (d_4_lastTokenWasLt_):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                                d_4_lastTokenWasLt_ = False
                                d_23_og3_: _dafny.Seq
                                d_24_oi3_: bool
                                d_25_oc3_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_23_og3_ = out13_
                                d_24_oi3_ = out14_
                                d_25_oc3_ = out15_
                                generated = d_23_og3_
                                insideConstrainedOut = d_24_oi3_
                                currentConstrainedOut = d_25_oc3_
                                d_26_rem3_: int
                                d_26_rem3_ = (maxSteps) - (d_1_steps_)
                                if (d_26_rem3_) > (0):
                                    d_27_cb3_: int
                                    if (d_26_rem3_) < (d_2_spanBudget_):
                                        d_27_cb3_ = d_26_rem3_
                                    elif True:
                                        d_27_cb3_ = d_2_spanBudget_
                                    d_28_cg3_: _dafny.Seq
                                    d_29_ci3_: bool
                                    d_30_cc3_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_cb3_)
                                    d_28_cg3_ = out16_
                                    d_29_ci3_ = out17_
                                    d_30_cc3_ = out18_
                                    generated = d_28_cg3_
                                    insideConstrainedOut = d_29_ci3_
                                    currentConstrainedOut = d_30_cc3_
                                    d_1_steps_ = (d_1_steps_) + (d_27_cb3_)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                                d_4_lastTokenWasLt_ = (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")))
                    elif True:
                        d_31_remainingSteps_: int
                        d_31_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_31_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_32_closeBudgetInner_: int
                        if (d_31_remainingSteps_) < (d_2_spanBudget_):
                            d_32_closeBudgetInner_ = d_31_remainingSteps_
                        elif True:
                            d_32_closeBudgetInner_ = d_2_spanBudget_
                        if (d_32_closeBudgetInner_) > (0):
                            d_33_cgI_: _dafny.Seq
                            d_34_ciI_: bool
                            d_35_ccI_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudgetInner_)
                            d_33_cgI_ = out19_
                            d_34_ciI_ = out20_
                            d_35_ccI_ = out21_
                            generated = d_33_cgI_
                            insideConstrainedOut = d_34_ciI_
                            currentConstrainedOut = d_35_ccI_
                            d_1_steps_ = (d_1_steps_) + (d_32_closeBudgetInner_)
                        elif True:
                            raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_36_remainingA_: int
            d_36_remainingA_ = (maxSteps) - (d_1_steps_)
            d_37_closeBudgetA_: int
            if (d_36_remainingA_) < (50):
                d_37_closeBudgetA_ = d_36_remainingA_
            elif True:
                d_37_closeBudgetA_ = 50
            if (d_37_closeBudgetA_) > (0):
                d_38_cgA_: _dafny.Seq
                d_39_ciA_: bool
                d_40_ccA_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: bool
                out24_: _dafny.Seq
                out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_37_closeBudgetA_)
                d_38_cgA_ = out22_
                d_39_ciA_ = out23_
                d_40_ccA_ = out24_
                generated = d_38_cgA_
                insideConstrainedOut = d_39_ciA_
                currentConstrainedOut = d_40_ccA_
                d_1_steps_ = (d_1_steps_) + (d_37_closeBudgetA_)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_41_genStr_: _dafny.Seq
            d_41_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
            d_42_openCount_: int
            d_42_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_41_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            if (d_42_openCount_) == (0):
                d_43_remainingB_: int
                d_43_remainingB_ = (maxSteps) - (d_1_steps_)
                if (d_43_remainingB_) >= (5):
                    d_44_ogB_: _dafny.Seq
                    d_45_oiB_: bool
                    d_46_ocB_: _dafny.Seq
                    out25_: _dafny.Seq
                    out26_: bool
                    out27_: _dafny.Seq
                    out25_, out26_, out27_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_44_ogB_ = out25_
                    d_45_oiB_ = out26_
                    d_46_ocB_ = out27_
                    generated = d_44_ogB_
                    insideConstrainedOut = d_45_oiB_
                    currentConstrainedOut = d_46_ocB_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_1_steps_) < (maxSteps):
                        d_47_remainingB2_: int
                        d_47_remainingB2_ = (maxSteps) - (d_1_steps_)
                        d_48_closeBudgetB_: int
                        if (d_47_remainingB2_) < (80):
                            d_48_closeBudgetB_ = d_47_remainingB2_
                        elif True:
                            d_48_closeBudgetB_ = 80
                        if (d_48_closeBudgetB_) > (0):
                            d_49_cgB_: _dafny.Seq
                            d_50_ciB_: bool
                            d_51_ccB_: _dafny.Seq
                            out28_: _dafny.Seq
                            out29_: bool
                            out30_: _dafny.Seq
                            out28_, out29_, out30_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_48_closeBudgetB_)
                            d_49_cgB_ = out28_
                            d_50_ciB_ = out29_
                            d_51_ccB_ = out30_
                            generated = d_49_cgB_
                            insideConstrainedOut = d_50_ciB_
                            currentConstrainedOut = d_51_ccB_
                            d_1_steps_ = (d_1_steps_) + (d_48_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

