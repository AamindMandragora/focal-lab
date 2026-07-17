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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write ONLY the final symbolic expression inside << >> using plain variable names (no curly braces: write x not {x}, write n1 not {n1}). Use operators +, -, *, /, //, %, int(). Use int() when the answer is a count or whole number. Include ALL relevant variables - do not simplify away any variable that affects the answer. Examples: <<n1 + n2>>, <<int(a * b / c)>>, <<(a + b) // c>>, <<int(n * p1 * p2 / 10000)>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanBudget_: int
        d_3_spanBudget_ = 60
        d_4_nearBudgetThreshold_: int
        d_4_nearBudgetThreshold_ = 120
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
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_13_remainingSteps_: int
                        d_13_remainingSteps_ = (maxSteps) - (d_1_steps_)
                        if (d_13_remainingSteps_) == (0):
                            raise _dafny.Break("0")
                        d_14_closeBudget2_: int
                        if (d_13_remainingSteps_) < (40):
                            d_14_closeBudget2_ = d_13_remainingSteps_
                        elif True:
                            d_14_closeBudget2_ = 40
                        d_15_cg2_: _dafny.Seq
                        d_16_ci2_: bool
                        d_17_cc2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget2_)
                        d_15_cg2_ = out7_
                        d_16_ci2_ = out8_
                        d_17_cc2_ = out9_
                        generated = d_15_cg2_
                        insideConstrainedOut = d_16_ci2_
                        currentConstrainedOut = d_17_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_14_closeBudget2_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_19_next_ = out10_
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
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget3_)
                            d_22_cg3_ = out11_
                            d_23_ci3_ = out12_
                            d_24_cc3_ = out13_
                            generated = d_22_cg3_
                            insideConstrainedOut = d_23_ci3_
                            currentConstrainedOut = d_24_cc3_
                            d_1_steps_ = (d_1_steps_) + (d_21_closeBudget3_)
                            raise _dafny.Break("0")
                        elif True:
                            d_25_isComplete_: bool
                            d_25_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_25_isComplete_:
                                d_26_remainingSteps_: int
                                d_26_remainingSteps_ = (maxSteps) - (d_1_steps_)
                                if (d_26_remainingSteps_) == (0):
                                    raise _dafny.Break("0")
                                d_27_closeBudget4_: int
                                if (d_26_remainingSteps_) < (25):
                                    d_27_closeBudget4_ = d_26_remainingSteps_
                                elif True:
                                    d_27_closeBudget4_ = 25
                                d_28_cg4_: _dafny.Seq
                                d_29_ci4_: bool
                                d_30_cc4_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget4_)
                                d_28_cg4_ = out14_
                                d_29_ci4_ = out15_
                                d_30_cc4_ = out16_
                                generated = d_28_cg4_
                                insideConstrainedOut = d_29_ci4_
                                currentConstrainedOut = d_30_cc4_
                                d_1_steps_ = (d_1_steps_) + (d_27_closeBudget4_)
                                d_2_spanSteps_ = 0
                            elif True:
                                d_31_ag_: _dafny.Seq
                                d_32_ai_: bool
                                d_33_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_31_ag_ = out17_
                                d_32_ai_ = out18_
                                d_33_ac_ = out19_
                                generated = d_31_ag_
                                insideConstrainedOut = d_32_ai_
                                currentConstrainedOut = d_33_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_34_remainingA_: int
            d_34_remainingA_ = (maxSteps) - (d_1_steps_)
            d_35_closeBudgetA_: int
            if (d_34_remainingA_) < (60):
                d_35_closeBudgetA_ = d_34_remainingA_
            elif True:
                d_35_closeBudgetA_ = 60
            d_36_cgA_: _dafny.Seq
            d_37_ciA_: bool
            d_38_ccA_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: bool
            out22_: _dafny.Seq
            out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_35_closeBudgetA_)
            d_36_cgA_ = out20_
            d_37_ciA_ = out21_
            d_38_ccA_ = out22_
            generated = d_36_cgA_
            insideConstrainedOut = d_37_ciA_
            currentConstrainedOut = d_38_ccA_
            d_1_steps_ = (d_1_steps_) + (d_35_closeBudgetA_)
        d_39_genStr_: _dafny.Seq
        d_39_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_40_openCount_: int
        d_40_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_39_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_40_openCount_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_41_remainingB_: int
            d_41_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_41_remainingB_) >= (5):
                d_42_ogB_: _dafny.Seq
                d_43_oiB_: bool
                d_44_ocB_: _dafny.Seq
                out23_: _dafny.Seq
                out24_: bool
                out25_: _dafny.Seq
                out23_, out24_, out25_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_42_ogB_ = out23_
                d_43_oiB_ = out24_
                d_44_ocB_ = out25_
                generated = d_42_ogB_
                insideConstrainedOut = d_43_oiB_
                currentConstrainedOut = d_44_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_45_remainingB2_: int
                    d_45_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_46_closeBudgetB_: int
                    if (d_45_remainingB2_) < (100):
                        d_46_closeBudgetB_ = d_45_remainingB2_
                    elif True:
                        d_46_closeBudgetB_ = 100
                    if (d_46_closeBudgetB_) > (0):
                        d_47_cgB_: _dafny.Seq
                        d_48_ciB_: bool
                        d_49_ccB_: _dafny.Seq
                        out26_: _dafny.Seq
                        out27_: bool
                        out28_: _dafny.Seq
                        out26_, out27_, out28_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_46_closeBudgetB_)
                        d_47_cgB_ = out26_
                        d_48_ciB_ = out27_
                        d_49_ccB_ = out28_
                        generated = d_47_cgB_
                        insideConstrainedOut = d_48_ciB_
                        currentConstrainedOut = d_49_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_46_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

