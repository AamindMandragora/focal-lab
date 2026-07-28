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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the end, write the final answer inside <<expression>> using plain variable names (no curly braces), numbers, and operators +, -, *, /, //, %, int(). Do not use ** or ^. Example final answer: <<n * price>>, <<int((a + b) // c)>>, <<(x - y) * z>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanBudget_: int
        d_2_spanBudget_ = 50
        d_3_forceOpenThreshold_: int
        d_3_forceOpenThreshold_ = 120
        d_4_finalSpanBudget_: int
        d_4_finalSpanBudget_ = 115
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_5_remainingBudget_: int
                    d_5_remainingBudget_ = (maxSteps) - (d_1_steps_)
                    if (d_5_remainingBudget_) <= (d_3_forceOpenThreshold_):
                        raise _dafny.Break("0")
                    if not(insideConstrainedOut):
                        d_6_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_7_og_: _dafny.Seq
                                d_8_oi_: bool
                                d_9_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_7_og_ = out1_
                                d_8_oi_ = out2_
                                d_9_oc_ = out3_
                                generated = d_7_og_
                                insideConstrainedOut = d_8_oi_
                                currentConstrainedOut = d_9_oc_
                                d_10_rem_: int
                                d_10_rem_ = (maxSteps) - (d_1_steps_)
                                if (d_10_rem_) > (0):
                                    d_11_cb_: int
                                    if (d_10_rem_) < (d_2_spanBudget_):
                                        d_11_cb_ = d_10_rem_
                                    elif True:
                                        d_11_cb_ = d_2_spanBudget_
                                    d_12_cg_: _dafny.Seq
                                    d_13_ci_: bool
                                    d_14_cc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_cb_)
                                    d_12_cg_ = out4_
                                    d_13_ci_ = out5_
                                    d_14_cc_ = out6_
                                    generated = d_12_cg_
                                    insideConstrainedOut = d_13_ci_
                                    currentConstrainedOut = d_14_cc_
                                    d_1_steps_ = (d_1_steps_) + (d_11_cb_)
                    elif True:
                        d_15_rem2_: int
                        d_15_rem2_ = (maxSteps) - (d_1_steps_)
                        if (d_15_rem2_) <= (d_3_forceOpenThreshold_):
                            raise _dafny.Break("0")
                        d_16_cb2_: int
                        if (d_15_rem2_) < (d_2_spanBudget_):
                            d_16_cb2_ = d_15_rem2_
                        elif True:
                            d_16_cb2_ = d_2_spanBudget_
                        if (d_16_cb2_) > (0):
                            d_17_cg2_: _dafny.Seq
                            d_18_ci2_: bool
                            d_19_cc2_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_cb2_)
                            d_17_cg2_ = out7_
                            d_18_ci2_ = out8_
                            d_19_cc2_ = out9_
                            generated = d_17_cg2_
                            insideConstrainedOut = d_18_ci2_
                            currentConstrainedOut = d_19_cc2_
                            d_1_steps_ = (d_1_steps_) + (d_16_cb2_)
                        elif True:
                            raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_20_remainingA_: int
            d_20_remainingA_ = (maxSteps) - (d_1_steps_)
            d_21_closeBudgetA_: int
            if (d_20_remainingA_) < (50):
                d_21_closeBudgetA_ = d_20_remainingA_
            elif True:
                d_21_closeBudgetA_ = 50
            if (d_21_closeBudgetA_) > (0):
                d_22_cgA_: _dafny.Seq
                d_23_ciA_: bool
                d_24_ccA_: _dafny.Seq
                out10_: _dafny.Seq
                out11_: bool
                out12_: _dafny.Seq
                out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudgetA_)
                d_22_cgA_ = out10_
                d_23_ciA_ = out11_
                d_24_ccA_ = out12_
                generated = d_22_cgA_
                insideConstrainedOut = d_23_ciA_
                currentConstrainedOut = d_24_ccA_
                d_1_steps_ = (d_1_steps_) + (d_21_closeBudgetA_)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_25_remainingB_: int
            d_25_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_25_remainingB_) >= (5):
                d_26_ogB_: _dafny.Seq
                d_27_oiB_: bool
                d_28_ocB_: _dafny.Seq
                out13_: _dafny.Seq
                out14_: bool
                out15_: _dafny.Seq
                out13_, out14_, out15_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_26_ogB_ = out13_
                d_27_oiB_ = out14_
                d_28_ocB_ = out15_
                generated = d_26_ogB_
                insideConstrainedOut = d_27_oiB_
                currentConstrainedOut = d_28_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_29_remainingB2_: int
                    d_29_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_30_closeBudgetB_: int
                    if (d_29_remainingB2_) < (d_4_finalSpanBudget_):
                        d_30_closeBudgetB_ = d_29_remainingB2_
                    elif True:
                        d_30_closeBudgetB_ = d_4_finalSpanBudget_
                    if (d_30_closeBudgetB_) > (0):
                        d_31_cgB_: _dafny.Seq
                        d_32_ciB_: bool
                        d_33_ccB_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudgetB_)
                        d_31_cgB_ = out16_
                        d_32_ciB_ = out17_
                        d_33_ccB_ = out18_
                        generated = d_31_cgB_
                        insideConstrainedOut = d_32_ciB_
                        currentConstrainedOut = d_33_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_30_closeBudgetB_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_34_remainingC_: int
            d_34_remainingC_ = (maxSteps) - (d_1_steps_)
            d_35_closeBudgetC_: int
            if (d_34_remainingC_) < (30):
                d_35_closeBudgetC_ = d_34_remainingC_
            elif True:
                d_35_closeBudgetC_ = 30
            if (d_35_closeBudgetC_) > (0):
                d_36_cgC_: _dafny.Seq
                d_37_ciC_: bool
                d_38_ccC_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_35_closeBudgetC_)
                d_36_cgC_ = out19_
                d_37_ciC_ = out20_
                d_38_ccC_ = out21_
                generated = d_36_cgC_
                insideConstrainedOut = d_37_ciC_
                currentConstrainedOut = d_38_ccC_
                d_1_steps_ = (d_1_steps_) + (d_35_closeBudgetC_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

