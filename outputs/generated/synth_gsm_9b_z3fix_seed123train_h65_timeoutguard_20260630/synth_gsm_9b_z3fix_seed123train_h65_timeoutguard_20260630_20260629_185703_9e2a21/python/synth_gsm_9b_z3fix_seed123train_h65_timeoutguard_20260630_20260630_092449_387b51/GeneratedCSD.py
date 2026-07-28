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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using only the variable names given (no curly braces). At the very end, write the final numeric expression inside << >> using ONLY variables from the problem and operators +, -, *, /, //, %, int(). The expression should directly compute the answer. Do NOT include extra variables or currency symbols. Example final lines: <<n1 + n2 * rate>>, <<int(total * percent / 100)>>, <<(a + b) // c>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_nearBudgetThreshold_: int
        d_2_nearBudgetThreshold_ = 120
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remainingBudget_: int
                        d_3_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_3_remainingBudget_) <= (3):
                            raise _dafny.Break("0")
                        elif (d_3_remainingBudget_) <= (d_2_nearBudgetThreshold_):
                            d_4_genStr_: _dafny.Seq
                            d_4_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
                            d_5_openCount_: int
                            d_5_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_4_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_6_closeCount_: int
                            d_6_closeCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_4_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                            if ((d_5_openCount_) > (0)) and ((d_6_closeCount_) >= (d_5_openCount_)):
                                raise _dafny.Break("0")
                            elif True:
                                d_7_og_: _dafny.Seq
                                d_8_oi_: bool
                                d_9_oc_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_7_og_ = out0_
                                d_8_oi_ = out1_
                                d_9_oc_ = out2_
                                generated = d_7_og_
                                insideConstrainedOut = d_8_oi_
                                currentConstrainedOut = d_9_oc_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_1_steps_) < (maxSteps):
                                    d_10_rem2_: int
                                    d_10_rem2_ = (maxSteps) - (d_1_steps_)
                                    d_11_closeBudget_: int
                                    if (d_10_rem2_) < ((d_2_nearBudgetThreshold_) - (1)):
                                        d_11_closeBudget_ = d_10_rem2_
                                    elif True:
                                        d_11_closeBudget_ = (d_2_nearBudgetThreshold_) - (1)
                                    d_12_cg_: _dafny.Seq
                                    d_13_ci_: bool
                                    d_14_cc_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out4_: bool
                                    out5_: _dafny.Seq
                                    out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget_)
                                    d_12_cg_ = out3_
                                    d_13_ci_ = out4_
                                    d_14_cc_ = out5_
                                    generated = d_12_cg_
                                    insideConstrainedOut = d_13_ci_
                                    currentConstrainedOut = d_14_cc_
                                    d_1_steps_ = (d_1_steps_) + (d_11_closeBudget_)
                        elif True:
                            d_15_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_15_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                    elif True:
                        d_16_remainingSteps_: int
                        d_16_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_16_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_17_closeBudget2_: int
                        if (d_16_remainingSteps_) < (40):
                            d_17_closeBudget2_ = d_16_remainingSteps_
                        elif True:
                            d_17_closeBudget2_ = 40
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
                        d_1_steps_ = (d_1_steps_) + (d_17_closeBudget2_)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_21_remainingA_: int
            d_21_remainingA_ = (maxSteps) - (d_1_steps_)
            d_22_closeBudgetA_: int
            if (d_21_remainingA_) < (60):
                d_22_closeBudgetA_ = d_21_remainingA_
            elif True:
                d_22_closeBudgetA_ = 60
            d_23_cgA_: _dafny.Seq
            d_24_ciA_: bool
            d_25_ccA_: _dafny.Seq
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudgetA_)
            d_23_cgA_ = out10_
            d_24_ciA_ = out11_
            d_25_ccA_ = out12_
            generated = d_23_cgA_
            insideConstrainedOut = d_24_ciA_
            currentConstrainedOut = d_25_ccA_
            d_1_steps_ = (d_1_steps_) + (d_22_closeBudgetA_)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_26_genStr2_: _dafny.Seq
            d_26_genStr2_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
            d_27_openCount2_: int
            d_27_openCount2_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_26_genStr2_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            if (d_27_openCount2_) == (0):
                d_28_remainingB_: int
                d_28_remainingB_ = (maxSteps) - (d_1_steps_)
                if (d_28_remainingB_) >= (5):
                    d_29_ogB_: _dafny.Seq
                    d_30_oiB_: bool
                    d_31_ocB_: _dafny.Seq
                    out13_: _dafny.Seq
                    out14_: bool
                    out15_: _dafny.Seq
                    out13_, out14_, out15_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_29_ogB_ = out13_
                    d_30_oiB_ = out14_
                    d_31_ocB_ = out15_
                    generated = d_29_ogB_
                    insideConstrainedOut = d_30_oiB_
                    currentConstrainedOut = d_31_ocB_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_1_steps_) < (maxSteps):
                        d_32_remainingB2_: int
                        d_32_remainingB2_ = (maxSteps) - (d_1_steps_)
                        d_33_closeBudgetB_: int
                        if (d_32_remainingB2_) < (100):
                            d_33_closeBudgetB_ = d_32_remainingB2_
                        elif True:
                            d_33_closeBudgetB_ = 100
                        if (d_33_closeBudgetB_) > (0):
                            d_34_cgB_: _dafny.Seq
                            d_35_ciB_: bool
                            d_36_ccB_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudgetB_)
                            d_34_cgB_ = out16_
                            d_35_ciB_ = out17_
                            d_36_ccB_ = out18_
                            generated = d_34_cgB_
                            insideConstrainedOut = d_35_ciB_
                            currentConstrainedOut = d_36_ccB_
                            d_1_steps_ = (d_1_steps_) + (d_33_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

