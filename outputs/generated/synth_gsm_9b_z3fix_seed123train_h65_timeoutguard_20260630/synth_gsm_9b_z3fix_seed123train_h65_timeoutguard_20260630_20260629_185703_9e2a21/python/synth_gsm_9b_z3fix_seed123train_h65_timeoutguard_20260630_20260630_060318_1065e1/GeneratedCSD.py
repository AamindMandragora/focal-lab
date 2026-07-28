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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write ONLY the final arithmetic expression as <<expression>> using plain variable names (NO curly braces: write n not {n}), numbers, and operators +, -, *, /, //, %, int(). The expression must be a valid arithmetic expression. Example: <<n1 + n2>>, <<int(a * b / c)>>, <<(a + b) // c>>. Do NOT write reasoning inside << >>. Do NOT use { or } inside << >>.")))
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
                        if (d_4_remainingBudget_) <= (120):
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
                            d_2_spanSteps_ = 0
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_9_og2_: _dafny.Seq
                                    d_10_oi2_: bool
                                    d_11_oc2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_9_og2_ = out4_
                                    d_10_oi2_ = out5_
                                    d_11_oc2_ = out6_
                                    generated = d_9_og2_
                                    insideConstrainedOut = d_10_oi2_
                                    currentConstrainedOut = d_11_oc2_
                                    d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_12_remainingForce_: int
                        d_12_remainingForce_ = (maxSteps) - (d_1_steps_)
                        if (d_12_remainingForce_) == (0):
                            raise _dafny.Break("0")
                        d_13_closeBudgetForce_: int
                        if (d_12_remainingForce_) < (40):
                            d_13_closeBudgetForce_ = d_12_remainingForce_
                        elif True:
                            d_13_closeBudgetForce_ = 40
                        d_14_cgF_: _dafny.Seq
                        d_15_ciF_: bool
                        d_16_ccF_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_closeBudgetForce_)
                        d_14_cgF_ = out7_
                        d_15_ciF_ = out8_
                        d_16_ccF_ = out9_
                        generated = d_14_cgF_
                        insideConstrainedOut = d_15_ciF_
                        currentConstrainedOut = d_16_ccF_
                        d_1_steps_ = (d_1_steps_) + (d_13_closeBudgetForce_)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_18_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_18_next_) == (eosToken):
                            d_19_remainingEos_: int
                            d_19_remainingEos_ = (maxSteps) - (d_1_steps_)
                            if (d_19_remainingEos_) == (0):
                                raise _dafny.Break("0")
                            d_20_closeBudgetEos_: int
                            if (d_19_remainingEos_) < (30):
                                d_20_closeBudgetEos_ = d_19_remainingEos_
                            elif True:
                                d_20_closeBudgetEos_ = 30
                            d_21_cgE_: _dafny.Seq
                            d_22_ciE_: bool
                            d_23_ccE_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudgetEos_)
                            d_21_cgE_ = out11_
                            d_22_ciE_ = out12_
                            d_23_ccE_ = out13_
                            generated = d_21_cgE_
                            insideConstrainedOut = d_22_ciE_
                            currentConstrainedOut = d_23_ccE_
                            d_1_steps_ = (d_1_steps_) + (d_20_closeBudgetEos_)
                            d_2_spanSteps_ = 0
                            raise _dafny.Break("0")
                        elif True:
                            d_24_isComplete_: bool
                            d_24_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_24_isComplete_:
                                d_25_remainingCo_: int
                                d_25_remainingCo_ = (maxSteps) - (d_1_steps_)
                                if (d_25_remainingCo_) == (0):
                                    raise _dafny.Break("0")
                                d_26_closeBudgetCo_: int
                                if (d_25_remainingCo_) < (20):
                                    d_26_closeBudgetCo_ = d_25_remainingCo_
                                elif True:
                                    d_26_closeBudgetCo_ = 20
                                d_27_cgCo_: _dafny.Seq
                                d_28_ciCo_: bool
                                d_29_ccCo_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudgetCo_)
                                d_27_cgCo_ = out14_
                                d_28_ciCo_ = out15_
                                d_29_ccCo_ = out16_
                                generated = d_27_cgCo_
                                insideConstrainedOut = d_28_ciCo_
                                currentConstrainedOut = d_29_ccCo_
                                d_1_steps_ = (d_1_steps_) + (d_26_closeBudgetCo_)
                                d_2_spanSteps_ = 0
                            elif True:
                                d_30_ag_: _dafny.Seq
                                d_31_ai_: bool
                                d_32_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                d_30_ag_ = out17_
                                d_31_ai_ = out18_
                                d_32_ac_ = out19_
                                generated = d_30_ag_
                                insideConstrainedOut = d_31_ai_
                                currentConstrainedOut = d_32_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_33_remainingA_: int
            d_33_remainingA_ = (maxSteps) - (d_1_steps_)
            d_34_closeBudgetA_: int
            if (d_33_remainingA_) < (50):
                d_34_closeBudgetA_ = d_33_remainingA_
            elif True:
                d_34_closeBudgetA_ = 50
            d_35_cgA_: _dafny.Seq
            d_36_ciA_: bool
            d_37_ccA_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: bool
            out22_: _dafny.Seq
            out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_34_closeBudgetA_)
            d_35_cgA_ = out20_
            d_36_ciA_ = out21_
            d_37_ccA_ = out22_
            generated = d_35_cgA_
            insideConstrainedOut = d_36_ciA_
            currentConstrainedOut = d_37_ccA_
            d_1_steps_ = (d_1_steps_) + (d_34_closeBudgetA_)
        d_38_genStr_: _dafny.Seq
        d_38_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_39_openCount_: int
        d_39_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_38_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_39_openCount_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_40_remainingB_: int
            d_40_remainingB_ = (maxSteps) - (d_1_steps_)
            if (d_40_remainingB_) >= (5):
                d_41_ogB_: _dafny.Seq
                d_42_oiB_: bool
                d_43_ocB_: _dafny.Seq
                out23_: _dafny.Seq
                out24_: bool
                out25_: _dafny.Seq
                out23_, out24_, out25_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_41_ogB_ = out23_
                d_42_oiB_ = out24_
                d_43_ocB_ = out25_
                generated = d_41_ogB_
                insideConstrainedOut = d_42_oiB_
                currentConstrainedOut = d_43_ocB_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_44_remainingB2_: int
                    d_44_remainingB2_ = (maxSteps) - (d_1_steps_)
                    d_45_closeBudgetB_: int
                    if (d_44_remainingB2_) < (90):
                        d_45_closeBudgetB_ = d_44_remainingB2_
                    elif True:
                        d_45_closeBudgetB_ = 90
                    if (d_45_closeBudgetB_) > (0):
                        d_46_cgB_: _dafny.Seq
                        d_47_ciB_: bool
                        d_48_ccB_: _dafny.Seq
                        out26_: _dafny.Seq
                        out27_: bool
                        out28_: _dafny.Seq
                        out26_, out27_, out28_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_45_closeBudgetB_)
                        d_46_cgB_ = out26_
                        d_47_ciB_ = out27_
                        d_48_ccB_ = out28_
                        generated = d_46_cgB_
                        insideConstrainedOut = d_47_ciB_
                        currentConstrainedOut = d_48_ccB_
                        d_1_steps_ = (d_1_steps_) + (d_45_closeBudgetB_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

