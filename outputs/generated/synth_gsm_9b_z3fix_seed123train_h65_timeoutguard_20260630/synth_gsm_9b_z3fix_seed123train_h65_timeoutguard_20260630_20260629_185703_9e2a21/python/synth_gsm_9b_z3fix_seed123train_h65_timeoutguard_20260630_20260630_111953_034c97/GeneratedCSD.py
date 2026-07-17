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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Use plain variable names (no curly braces). Write the final answer as <<expression>>. Only use: variable names, numbers, +, -, *, /, //, %, int(), parentheses. No {}, no **.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanBudget_: int
        d_3_spanBudget_ = 60
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remainingBudget_: int
                        d_4_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_4_remainingBudget_) <= (3):
                            raise _dafny.Break("0")
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_6_og2_: _dafny.Seq
                                d_7_oi2_: bool
                                d_8_oc2_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_6_og2_ = out1_
                                d_7_oi2_ = out2_
                                d_8_oc2_ = out3_
                                generated = d_6_og2_
                                insideConstrainedOut = d_7_oi2_
                                currentConstrainedOut = d_8_oc2_
                                d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_9_remainingSteps_: int
                        d_9_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_9_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_10_closeBudget2_: int
                        if (d_9_remainingSteps_) < (40):
                            d_10_closeBudget2_ = d_9_remainingSteps_
                        elif True:
                            d_10_closeBudget2_ = 40
                        d_11_cg2_: _dafny.Seq
                        d_12_ci2_: bool
                        d_13_cc2_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_10_closeBudget2_)
                        d_11_cg2_ = out4_
                        d_12_ci2_ = out5_
                        d_13_cc2_ = out6_
                        generated = d_11_cg2_
                        insideConstrainedOut = d_12_ci2_
                        currentConstrainedOut = d_13_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_10_closeBudget2_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_14_cg1_: _dafny.Seq
                        d_15_ci1_: bool
                        d_16_cc1_: _dafny.Seq
                        d_17_closed1_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_14_cg1_ = out7_
                        d_15_ci1_ = out8_
                        d_16_cc1_ = out9_
                        d_17_closed1_ = out10_
                        if d_17_closed1_:
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_14_cg1_
                            insideConstrainedOut = d_15_ci1_
                            currentConstrainedOut = d_16_cc1_
                            d_2_spanSteps_ = 0
                        elif True:
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_19_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                            if (d_19_next_) == (eosToken):
                                d_20_remainingSteps_: int
                                d_20_remainingSteps_ = (maxSteps) - (d_1_steps_)
                                if (d_20_remainingSteps_) == (0):
                                    raise _dafny.Break("0")
                                d_21_closeBudget3_: int
                                if (d_20_remainingSteps_) < (30):
                                    d_21_closeBudget3_ = d_20_remainingSteps_
                                elif True:
                                    d_21_closeBudget3_ = 30
                                d_22_cg3_: _dafny.Seq
                                d_23_ci3_: bool
                                d_24_cc3_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget3_)
                                d_22_cg3_ = out12_
                                d_23_ci3_ = out13_
                                d_24_cc3_ = out14_
                                generated = d_22_cg3_
                                insideConstrainedOut = d_23_ci3_
                                currentConstrainedOut = d_24_cc3_
                                d_1_steps_ = (d_1_steps_) + (d_21_closeBudget3_)
                                d_2_spanSteps_ = 0
                                raise _dafny.Break("0")
                            elif True:
                                d_25_ag_: _dafny.Seq
                                d_26_ai_: bool
                                d_27_ac_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_25_ag_ = out15_
                                d_26_ai_ = out16_
                                d_27_ac_ = out17_
                                generated = d_25_ag_
                                insideConstrainedOut = d_26_ai_
                                currentConstrainedOut = d_27_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_28_remainingA_: int
            d_28_remainingA_ = (maxSteps) - (d_1_steps_)
            d_29_closeBudgetA_: int
            if (d_28_remainingA_) < (60):
                d_29_closeBudgetA_ = d_28_remainingA_
            elif True:
                d_29_closeBudgetA_ = 60
            d_30_cgA_: _dafny.Seq
            d_31_ciA_: bool
            d_32_ccA_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudgetA_)
            d_30_cgA_ = out18_
            d_31_ciA_ = out19_
            d_32_ccA_ = out20_
            generated = d_30_cgA_
            insideConstrainedOut = d_31_ciA_
            currentConstrainedOut = d_32_ccA_
            d_1_steps_ = (d_1_steps_) + (d_29_closeBudgetA_)
        d_33_genStr_: _dafny.Seq
        d_33_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_34_openCount_: int
        d_34_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_33_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_34_openCount_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_35_remainingB_: int
            d_35_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_35_remainingB_) >= (5):
                d_36_ogB_: _dafny.Seq
                d_37_oiB_: bool
                d_38_ocB_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_36_ogB_ = out21_
                d_37_oiB_ = out22_
                d_38_ocB_ = out23_
                generated = d_36_ogB_
                insideConstrainedOut = d_37_oiB_
                currentConstrainedOut = d_38_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_39_remainingB2_: int
                    d_39_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_40_closeBudgetB_: int
                    if (d_39_remainingB2_) < (80):
                        d_40_closeBudgetB_ = d_39_remainingB2_
                    elif True:
                        d_40_closeBudgetB_ = 80
                    if (d_40_closeBudgetB_) > (0):
                        d_41_cgB_: _dafny.Seq
                        d_42_ciB_: bool
                        d_43_ccB_: _dafny.Seq
                        out24_: _dafny.Seq
                        out25_: bool
                        out26_: _dafny.Seq
                        out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_40_closeBudgetB_)
                        d_41_cgB_ = out24_
                        d_42_ciB_ = out25_
                        d_43_ccB_ = out26_
                        generated = d_41_cgB_
                        insideConstrainedOut = d_42_ciB_
                        currentConstrainedOut = d_43_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_40_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

