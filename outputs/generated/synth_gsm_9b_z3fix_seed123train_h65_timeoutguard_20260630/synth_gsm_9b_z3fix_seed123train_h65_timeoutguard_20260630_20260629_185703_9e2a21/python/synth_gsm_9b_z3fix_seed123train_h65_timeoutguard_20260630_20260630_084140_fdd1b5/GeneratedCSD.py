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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show your reasoning, then write ONLY the final symbolic expression inside << >> delimiters. Use plain variable names without curly braces (write n1 not {n1}). Use only these operators: +, -, *, /, //, %, int(). Do NOT use ** for powers. For counts/integers use int(). Keep the final expression concise and accurate. Examples: <<n1 + n2>>, <<int(n * p / 100)>>, <<(a + b) // c>>, <<int(n * p1 * p2 / 10000)>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_nearBudgetThreshold_: int
        d_2_nearBudgetThreshold_ = 100
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remainingBudget_: int
                        d_3_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_3_remainingBudget_) <= (3):
                            raise _dafny.Break("0")
                        elif (d_3_remainingBudget_) <= (d_2_nearBudgetThreshold_):
                            d_4_og_: _dafny.Seq
                            d_5_oi_: bool
                            d_6_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_og_ = out0_
                            d_5_oi_ = out1_
                            d_6_oc_ = out2_
                            generated = d_4_og_
                            insideConstrainedOut = d_5_oi_
                            currentConstrainedOut = d_6_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_1_steps_) < (maxSteps):
                                d_7_rem2_: int
                                d_7_rem2_ = (maxSteps) - (d_1_steps_)
                                d_8_closeBudget_: int
                                if (d_7_rem2_) < (80):
                                    d_8_closeBudget_ = d_7_rem2_
                                elif True:
                                    d_8_closeBudget_ = 80
                                d_9_cg_: _dafny.Seq
                                d_10_ci_: bool
                                d_11_cc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_closeBudget_)
                                d_9_cg_ = out3_
                                d_10_ci_ = out4_
                                d_11_cc_ = out5_
                                generated = d_9_cg_
                                insideConstrainedOut = d_10_ci_
                                currentConstrainedOut = d_11_cc_
                                d_1_steps_ = (d_1_steps_) + (d_8_closeBudget_)
                        elif True:
                            d_12_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_13_og2_: _dafny.Seq
                                    d_14_oi2_: bool
                                    d_15_oc2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_13_og2_ = out7_
                                    d_14_oi2_ = out8_
                                    d_15_oc2_ = out9_
                                    generated = d_13_og2_
                                    insideConstrainedOut = d_14_oi2_
                                    currentConstrainedOut = d_15_oc2_
                                    if (d_1_steps_) < (maxSteps):
                                        d_16_rem3_: int
                                        d_16_rem3_ = (maxSteps) - (d_1_steps_)
                                        d_17_closeBudget2_: int
                                        if (d_16_rem3_) < (80):
                                            d_17_closeBudget2_ = d_16_rem3_
                                        elif True:
                                            d_17_closeBudget2_ = 80
                                        d_18_cg2_: _dafny.Seq
                                        d_19_ci2_: bool
                                        d_20_cc2_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget2_)
                                        d_18_cg2_ = out10_
                                        d_19_ci2_ = out11_
                                        d_20_cc2_ = out12_
                                        generated = d_18_cg2_
                                        insideConstrainedOut = d_19_ci2_
                                        currentConstrainedOut = d_20_cc2_
                                        d_1_steps_ = (d_1_steps_) + (d_17_closeBudget2_)
                    elif True:
                        d_21_remainingSteps_: int
                        d_21_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_21_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_22_closeBudget3_: int
                        if (d_21_remainingSteps_) < (40):
                            d_22_closeBudget3_ = d_21_remainingSteps_
                        elif True:
                            d_22_closeBudget3_ = 40
                        d_23_cg3_: _dafny.Seq
                        d_24_ci3_: bool
                        d_25_cc3_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget3_)
                        d_23_cg3_ = out13_
                        d_24_ci3_ = out14_
                        d_25_cc3_ = out15_
                        generated = d_23_cg3_
                        insideConstrainedOut = d_24_ci3_
                        currentConstrainedOut = d_25_cc3_
                        d_1_steps_ = (d_1_steps_) + (d_22_closeBudget3_)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_26_remainingA_: int
            d_26_remainingA_ = (maxSteps) - (d_1_steps_)
            d_27_closeBudgetA_: int
            if (d_26_remainingA_) < (60):
                d_27_closeBudgetA_ = d_26_remainingA_
            elif True:
                d_27_closeBudgetA_ = 60
            d_28_cgA_: _dafny.Seq
            d_29_ciA_: bool
            d_30_ccA_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudgetA_)
            d_28_cgA_ = out16_
            d_29_ciA_ = out17_
            d_30_ccA_ = out18_
            generated = d_28_cgA_
            insideConstrainedOut = d_29_ciA_
            currentConstrainedOut = d_30_ccA_
            d_1_steps_ = (d_1_steps_) + (d_27_closeBudgetA_)
        d_31_genStr_: _dafny.Seq
        d_31_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_32_openCount_: int
        d_32_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_31_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_32_openCount_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_33_remainingB_: int
            d_33_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_33_remainingB_) >= (5):
                d_34_ogB_: _dafny.Seq
                d_35_oiB_: bool
                d_36_ocB_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_34_ogB_ = out19_
                d_35_oiB_ = out20_
                d_36_ocB_ = out21_
                generated = d_34_ogB_
                insideConstrainedOut = d_35_oiB_
                currentConstrainedOut = d_36_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_37_remainingB2_: int
                    d_37_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_38_closeBudgetB_: int
                    if (d_37_remainingB2_) < (100):
                        d_38_closeBudgetB_ = d_37_remainingB2_
                    elif True:
                        d_38_closeBudgetB_ = 100
                    if (d_38_closeBudgetB_) > (0):
                        d_39_cgB_: _dafny.Seq
                        d_40_ciB_: bool
                        d_41_ccB_: _dafny.Seq
                        out22_: _dafny.Seq
                        out23_: bool
                        out24_: _dafny.Seq
                        out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_38_closeBudgetB_)
                        d_39_cgB_ = out22_
                        d_40_ciB_ = out23_
                        d_41_ccB_ = out24_
                        generated = d_39_cgB_
                        insideConstrainedOut = d_40_ciB_
                        currentConstrainedOut = d_41_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_38_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

