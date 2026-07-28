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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. At the very end, write the final numeric expression as <<expression>> using the variable names from the problem (no curly braces), numbers, and operators: +, -, *, /, //, %, int(), **. Do not include any text inside << >>. Example: <<n1 + n2>>, <<int(a * b / c)>>, <<(a + b) // c>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_3_og_: _dafny.Seq
                                d_4_oi_: bool
                                d_5_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_3_og_ = out1_
                                d_4_oi_ = out2_
                                d_5_oc_ = out3_
                                generated = d_3_og_
                                insideConstrainedOut = d_4_oi_
                                currentConstrainedOut = d_5_oc_
                    elif True:
                        d_6_cg_: _dafny.Seq
                        d_7_ci_: bool
                        d_8_cc_: _dafny.Seq
                        d_9_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_6_cg_ = out4_
                        d_7_ci_ = out5_
                        d_8_cc_ = out6_
                        d_9_closed_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_9_closed_:
                            generated = d_6_cg_
                            insideConstrainedOut = d_7_ci_
                            currentConstrainedOut = d_8_cc_
                        elif True:
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_11_next_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_11_next_ = out8_
                            if (d_11_next_) == (eosToken):
                                d_12_remainingEos_: int
                                d_12_remainingEos_ = (maxSteps) - (d_1_steps_)
                                if (d_12_remainingEos_) == (0):
                                    raise _dafny.Break("0")
                                d_13_closeBudgetEos_: int
                                if (d_12_remainingEos_) < (30):
                                    d_13_closeBudgetEos_ = d_12_remainingEos_
                                elif True:
                                    d_13_closeBudgetEos_ = 30
                                d_14_cgE_: _dafny.Seq
                                d_15_ciE_: bool
                                d_16_ccE_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_closeBudgetEos_)
                                d_14_cgE_ = out9_
                                d_15_ciE_ = out10_
                                d_16_ccE_ = out11_
                                generated = d_14_cgE_
                                insideConstrainedOut = d_15_ciE_
                                currentConstrainedOut = d_16_ccE_
                                d_1_steps_ = (d_1_steps_) + (d_13_closeBudgetEos_)
                                raise _dafny.Break("0")
                            elif True:
                                d_17_isComplete_: bool
                                d_17_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_17_isComplete_:
                                    d_18_remainingCo_: int
                                    d_18_remainingCo_ = (maxSteps) - (d_1_steps_)
                                    if (d_18_remainingCo_) == (0):
                                        raise _dafny.Break("0")
                                    d_19_closeBudgetCo_: int
                                    if (d_18_remainingCo_) < (20):
                                        d_19_closeBudgetCo_ = d_18_remainingCo_
                                    elif True:
                                        d_19_closeBudgetCo_ = 20
                                    d_20_cgCo_: _dafny.Seq
                                    d_21_ciCo_: bool
                                    d_22_ccCo_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudgetCo_)
                                    d_20_cgCo_ = out12_
                                    d_21_ciCo_ = out13_
                                    d_22_ccCo_ = out14_
                                    generated = d_20_cgCo_
                                    insideConstrainedOut = d_21_ciCo_
                                    currentConstrainedOut = d_22_ccCo_
                                    d_1_steps_ = (d_1_steps_) + (d_19_closeBudgetCo_)
                                elif True:
                                    d_23_ag_: _dafny.Seq
                                    d_24_ai_: bool
                                    d_25_ac_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                    d_23_ag_ = out15_
                                    d_24_ai_ = out16_
                                    d_25_ac_ = out17_
                                    generated = d_23_ag_
                                    insideConstrainedOut = d_24_ai_
                                    currentConstrainedOut = d_25_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_26_remainingPost_: int
            d_26_remainingPost_ = (maxSteps) - (d_1_steps_)
            d_27_closeBudgetPost_: int
            if (d_26_remainingPost_) < (60):
                d_27_closeBudgetPost_ = d_26_remainingPost_
            elif True:
                d_27_closeBudgetPost_ = 60
            d_28_cgP_: _dafny.Seq
            d_29_ciP_: bool
            d_30_ccP_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudgetPost_)
            d_28_cgP_ = out18_
            d_29_ciP_ = out19_
            d_30_ccP_ = out20_
            generated = d_28_cgP_
            insideConstrainedOut = d_29_ciP_
            currentConstrainedOut = d_30_ccP_
            d_1_steps_ = (d_1_steps_) + (d_27_closeBudgetPost_)
        d_31_genStr_: _dafny.Seq
        d_31_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
        d_32_openCount_: int
        d_32_openCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_31_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        if (((d_32_openCount_) == (0)) and (not(insideConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
            d_33_remainingForce_: int
            d_33_remainingForce_ = (maxSteps) - (d_1_steps_)
            if (d_33_remainingForce_) >= (10):
                d_34_ogF_: _dafny.Seq
                d_35_oiF_: bool
                d_36_ocF_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_34_ogF_ = out21_
                d_35_oiF_ = out22_
                d_36_ocF_ = out23_
                generated = d_34_ogF_
                insideConstrainedOut = d_35_oiF_
                currentConstrainedOut = d_36_ocF_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_1_steps_) < (maxSteps):
                    d_37_remainingFill_: int
                    d_37_remainingFill_ = (maxSteps) - (d_1_steps_)
                    d_38_closeBudgetFill_: int
                    if (d_37_remainingFill_) < (80):
                        d_38_closeBudgetFill_ = d_37_remainingFill_
                    elif True:
                        d_38_closeBudgetFill_ = 80
                    if (d_38_closeBudgetFill_) > (0):
                        d_39_cgF2_: _dafny.Seq
                        d_40_ciF2_: bool
                        d_41_ccF2_: _dafny.Seq
                        out24_: _dafny.Seq
                        out25_: bool
                        out26_: _dafny.Seq
                        out24_, out25_, out26_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_38_closeBudgetFill_)
                        d_39_cgF2_ = out24_
                        d_40_ciF2_ = out25_
                        d_41_ccF2_ = out26_
                        generated = d_39_cgF2_
                        insideConstrainedOut = d_40_ciF2_
                        currentConstrainedOut = d_41_ccF2_
                        d_1_steps_ = (d_1_steps_) + (d_38_closeBudgetFill_)
        if (d_1_steps_) > (maxSteps):
            cost = maxSteps
        elif True:
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

