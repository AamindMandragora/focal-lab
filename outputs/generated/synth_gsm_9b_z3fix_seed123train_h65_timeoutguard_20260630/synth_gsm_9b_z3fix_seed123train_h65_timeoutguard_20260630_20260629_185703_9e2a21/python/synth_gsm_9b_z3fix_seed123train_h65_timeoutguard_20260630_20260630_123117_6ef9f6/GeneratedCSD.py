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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. At the very end, write your final answer expression inside << >> delimiters, using plain variable names without curly braces. Examples: <<n * p>>, <<int(a + b)>>, <<(x + y) // z>>. Write exactly one <<expression>> at the end.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_remainingBudget_: int
                        d_2_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_2_remainingBudget_) <= (5):
                            raise _dafny.Break("0")
                        d_3_genStr_: _dafny.Seq
                        d_3_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
                        d_4_openCount_: int
                        d_4_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_3_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        if ((d_2_remainingBudget_) <= (120)) and ((d_4_openCount_) == (0)):
                            d_5_og_: _dafny.Seq
                            d_6_oi_: bool
                            d_7_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_og_ = out0_
                            d_6_oi_ = out1_
                            d_7_oc_ = out2_
                            generated = d_5_og_
                            insideConstrainedOut = d_6_oi_
                            currentConstrainedOut = d_7_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_1_steps_) < (maxSteps):
                                d_8_rem2_: int
                                d_8_rem2_ = (maxSteps) - (d_1_steps_)
                                d_9_cb_: int
                                if (d_8_rem2_) < (80):
                                    d_9_cb_ = d_8_rem2_
                                elif True:
                                    d_9_cb_ = 80
                                d_10_cg2_: _dafny.Seq
                                d_11_ci2_: bool
                                d_12_cc2_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_cb_)
                                d_10_cg2_ = out3_
                                d_11_ci2_ = out4_
                                d_12_cc2_ = out5_
                                generated = d_10_cg2_
                                insideConstrainedOut = d_11_ci2_
                                currentConstrainedOut = d_12_cc2_
                                d_1_steps_ = (d_1_steps_) + (d_9_cb_)
                        elif True:
                            d_13_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_14_og2_: _dafny.Seq
                                    d_15_oi2_: bool
                                    d_16_oc2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_14_og2_ = out7_
                                    d_15_oi2_ = out8_
                                    d_16_oc2_ = out9_
                                    generated = d_14_og2_
                                    insideConstrainedOut = d_15_oi2_
                                    currentConstrainedOut = d_16_oc2_
                                    if (d_1_steps_) < (maxSteps):
                                        d_17_rem3_: int
                                        d_17_rem3_ = (maxSteps) - (d_1_steps_)
                                        d_18_cb3_: int
                                        if (d_17_rem3_) < (80):
                                            d_18_cb3_ = d_17_rem3_
                                        elif True:
                                            d_18_cb3_ = 80
                                        d_19_cg3_: _dafny.Seq
                                        d_20_ci3_: bool
                                        d_21_cc3_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_cb3_)
                                        d_19_cg3_ = out10_
                                        d_20_ci3_ = out11_
                                        d_21_cc3_ = out12_
                                        generated = d_19_cg3_
                                        insideConstrainedOut = d_20_ci3_
                                        currentConstrainedOut = d_21_cc3_
                                        d_1_steps_ = (d_1_steps_) + (d_18_cb3_)
                    elif True:
                        d_22_remainingBudget2_: int
                        d_22_remainingBudget2_ = (maxSteps) - (d_1_steps_)
                        if (d_22_remainingBudget2_) == (0):
                            raise _dafny.Break("0")
                        d_23_cb4_: int
                        if (d_22_remainingBudget2_) < (40):
                            d_23_cb4_ = d_22_remainingBudget2_
                        elif True:
                            d_23_cb4_ = 40
                        d_24_cg4_: _dafny.Seq
                        d_25_ci4_: bool
                        d_26_cc4_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_cb4_)
                        d_24_cg4_ = out13_
                        d_25_ci4_ = out14_
                        d_26_cc4_ = out15_
                        generated = d_24_cg4_
                        insideConstrainedOut = d_25_ci4_
                        currentConstrainedOut = d_26_cc4_
                        d_1_steps_ = (d_1_steps_) + (d_23_cb4_)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_27_remainingA_: int
            d_27_remainingA_ = (maxSteps) - (d_1_steps_)
            d_28_closeBudgetA_: int
            if (d_27_remainingA_) < (60):
                d_28_closeBudgetA_ = d_27_remainingA_
            elif True:
                d_28_closeBudgetA_ = 60
            d_29_cgA_: _dafny.Seq
            d_30_ciA_: bool
            d_31_ccA_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_closeBudgetA_)
            d_29_cgA_ = out16_
            d_30_ciA_ = out17_
            d_31_ccA_ = out18_
            generated = d_29_cgA_
            insideConstrainedOut = d_30_ciA_
            currentConstrainedOut = d_31_ccA_
            d_1_steps_ = (d_1_steps_) + (d_28_closeBudgetA_)
        d_32_genStrFinal_: _dafny.Seq
        d_32_genStrFinal_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_33_openCountFinal_: int
        d_33_openCountFinal_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_32_genStrFinal_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_33_openCountFinal_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_34_remainingB_: int
            d_34_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_34_remainingB_) >= (5):
                d_35_ogB_: _dafny.Seq
                d_36_oiB_: bool
                d_37_ocB_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_35_ogB_ = out19_
                d_36_oiB_ = out20_
                d_37_ocB_ = out21_
                generated = d_35_ogB_
                insideConstrainedOut = d_36_oiB_
                currentConstrainedOut = d_37_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_38_remainingB2_: int
                    d_38_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_39_closeBudgetB_: int
                    if (d_38_remainingB2_) < (90):
                        d_39_closeBudgetB_ = d_38_remainingB2_
                    elif True:
                        d_39_closeBudgetB_ = 90
                    if (d_39_closeBudgetB_) > (0):
                        d_40_cgB_: _dafny.Seq
                        d_41_ciB_: bool
                        d_42_ccB_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_39_closeBudgetB_)
                        d_40_cgB_ = out22_
                        d_41_ciB_ = out23_
                        d_42_ccB_ = out24_
                        generated = d_40_cgB_
                        insideConstrainedOut = d_41_ciB_
                        currentConstrainedOut = d_42_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_39_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

