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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_og_: _dafny.Seq
                                d_6_oi_: bool
                                d_7_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_5_og_ = out1_
                                d_6_oi_ = out2_
                                d_7_oc_ = out3_
                                generated = d_5_og_
                                insideConstrainedOut = d_6_oi_
                                currentConstrainedOut = d_7_oc_
                                d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_8_remainingSteps_: int
                        d_8_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_8_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_9_closeBudget2_: int
                        if (d_8_remainingSteps_) < (40):
                            d_9_closeBudget2_ = d_8_remainingSteps_
                        elif True:
                            d_9_closeBudget2_ = 40
                        d_10_cg2_: _dafny.Seq
                        d_11_ci2_: bool
                        d_12_cc2_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget2_)
                        d_10_cg2_ = out4_
                        d_11_ci2_ = out5_
                        d_12_cc2_ = out6_
                        generated = d_10_cg2_
                        insideConstrainedOut = d_11_ci2_
                        currentConstrainedOut = d_12_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_9_closeBudget2_)
                        d_2_spanSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_remainingSteps_: int
                        d_13_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_13_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_14_closeBudget_: int
                        if (d_13_remainingSteps_) < (20):
                            d_14_closeBudget_ = d_13_remainingSteps_
                        elif True:
                            d_14_closeBudget_ = 20
                        d_15_cg_: _dafny.Seq
                        d_16_ci_: bool
                        d_17_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
                        d_15_cg_ = out7_
                        d_16_ci_ = out8_
                        d_17_cc_ = out9_
                        generated = d_15_cg_
                        insideConstrainedOut = d_16_ci_
                        currentConstrainedOut = d_17_cc_
                        d_1_steps_ = (d_1_steps_) + (d_14_closeBudget_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        d_20_wasConstrained_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out10_, out11_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_19_next_ = out10_
                        d_20_wasConstrained_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_19_next_) == (eosToken):
                            d_21_remainingSteps_: int
                            d_21_remainingSteps_ = (maxSteps) - (d_1_steps_)
                            if (d_21_remainingSteps_) == (0):
                                raise _dafny.Break("0")
                            d_22_closeBudget3_: int
                            if (d_21_remainingSteps_) < (30):
                                d_22_closeBudget3_ = d_21_remainingSteps_
                            elif True:
                                d_22_closeBudget3_ = 30
                            d_23_cg3_: _dafny.Seq
                            d_24_ci3_: bool
                            d_25_cc3_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget3_)
                            d_23_cg3_ = out12_
                            d_24_ci3_ = out13_
                            d_25_cc3_ = out14_
                            generated = d_23_cg3_
                            insideConstrainedOut = d_24_ci3_
                            currentConstrainedOut = d_25_cc3_
                            d_1_steps_ = (d_1_steps_) + (d_22_closeBudget3_)
                            raise _dafny.Break("0")
                        elif True:
                            d_26_ag_: _dafny.Seq
                            d_27_ai_: bool
                            d_28_ac_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_26_ag_ = out15_
                            d_27_ai_ = out16_
                            d_28_ac_ = out17_
                            generated = d_26_ag_
                            insideConstrainedOut = d_27_ai_
                            currentConstrainedOut = d_28_ac_
                    pass
            pass
        if insideConstrainedOut:
            d_29_remainingA_: int
            d_29_remainingA_ = (maxSteps) - (d_1_steps_)
            if (d_29_remainingA_) > (0):
                d_30_closeBudgetA_: int
                if (d_29_remainingA_) < (50):
                    d_30_closeBudgetA_ = d_29_remainingA_
                elif True:
                    d_30_closeBudgetA_ = 50
                d_31_cgA_: _dafny.Seq
                d_32_ciA_: bool
                d_33_ccA_: _dafny.Seq
                out18_: _dafny.Seq
                out19_: bool
                out20_: _dafny.Seq
                out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudgetA_)
                d_31_cgA_ = out18_
                d_32_ciA_ = out19_
                d_33_ccA_ = out20_
                generated = d_31_cgA_
                insideConstrainedOut = d_32_ciA_
                currentConstrainedOut = d_33_ccA_
                d_1_steps_ = (d_1_steps_) + (d_30_closeBudgetA_)
        d_34_genStr_: _dafny.Seq
        d_34_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_35_openCount_: int
        d_35_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_34_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if ((d_35_openCount_) == (0)) and (not(insideConstrainedOut)):
            d_36_remainingB_: int
            d_36_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_36_remainingB_) >= (5):
                d_37_ogB_: _dafny.Seq
                d_38_oiB_: bool
                d_39_ocB_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_37_ogB_ = out21_
                d_38_oiB_ = out22_
                d_39_ocB_ = out23_
                generated = d_37_ogB_
                insideConstrainedOut = d_38_oiB_
                currentConstrainedOut = d_39_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                d_40_remainingB2_: int
                d_40_remainingB2_ = (maxSteps) - (d_1_steps_)
                if (d_40_remainingB2_) > (0):
                    d_41_closeBudgetB_: int
                    if (d_40_remainingB2_) < (80):
                        d_41_closeBudgetB_ = d_40_remainingB2_
                    elif True:
                        d_41_closeBudgetB_ = 80
                    if (d_41_closeBudgetB_) > (0):
                        d_42_cgB_: _dafny.Seq
                        d_43_ciB_: bool
                        d_44_ccB_: _dafny.Seq
                        out24_: _dafny.Seq
                        out25_: bool
                        out26_: _dafny.Seq
                        out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_41_closeBudgetB_)
                        d_42_cgB_ = out24_
                        d_43_ciB_ = out25_
                        d_44_ccB_ = out26_
                        generated = d_42_cgB_
                        insideConstrainedOut = d_43_ciB_
                        currentConstrainedOut = d_44_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_41_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

