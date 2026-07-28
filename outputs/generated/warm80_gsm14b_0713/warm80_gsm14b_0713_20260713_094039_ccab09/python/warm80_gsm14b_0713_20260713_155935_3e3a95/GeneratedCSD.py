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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the given variable names. At the end, write: The final answer is <<ANSWER>> where ANSWER is an arithmetic expression using only the variable names, numbers, +, -, *, /, (, ), and ^ for powers."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        if (maxSteps) >= (10):
            d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        elif True:
            d_4_freeStepsTarget_ = _dafny.euclidian_division(maxSteps, 2)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_seenCompleteSpan_: bool
        d_6_seenCompleteSpan_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_budgetLeft_: int
                        d_7_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        d_8_shouldForce_: bool
                        d_8_shouldForce_ = ((not(d_5_forcedFinalSpan_)) and (not(d_6_seenCompleteSpan_))) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or ((d_7_budgetLeft_) <= (20)))
                        if (d_8_shouldForce_) and ((d_7_budgetLeft_) >= (5)):
                            d_9_og_: _dafny.Seq
                            d_10_oi_: bool
                            d_11_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_og_ = out0_
                            d_10_oi_ = out1_
                            d_11_oc_ = out2_
                            generated = d_9_og_
                            insideConstrainedOut = d_10_oi_
                            currentConstrainedOut = d_11_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_5_forcedFinalSpan_ = True
                        elif (d_8_shouldForce_) and ((d_7_budgetLeft_) < (5)):
                            raise _dafny.Break("0")
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                if ((not(d_6_seenCompleteSpan_)) and (not(d_5_forcedFinalSpan_))) and (((maxSteps) - (d_2_steps_)) >= (5)):
                                    d_13_og_: _dafny.Seq
                                    d_14_oi_: bool
                                    d_15_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_13_og_ = out4_
                                    d_14_oi_ = out5_
                                    d_15_oc_ = out6_
                                    generated = d_13_og_
                                    insideConstrainedOut = d_14_oi_
                                    currentConstrainedOut = d_15_oc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_5_forcedFinalSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_16_budgetLeft2_: int
                                    d_16_budgetLeft2_ = (maxSteps) - (d_2_steps_)
                                    if (d_16_budgetLeft2_) >= (3):
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out9_: _dafny.Seq
                                        out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                        generated = out7_
                                        insideConstrainedOut = out8_
                                        currentConstrainedOut = out9_
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_17_cg_: _dafny.Seq
                            d_18_ci_: bool
                            d_19_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_17_cg_ = out10_
                            d_18_ci_ = out11_
                            d_19_cc_ = out12_
                            generated = d_17_cg_
                            insideConstrainedOut = d_18_ci_
                            currentConstrainedOut = d_19_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_6_seenCompleteSpan_ = True
                            if d_5_forcedFinalSpan_:
                                raise _dafny.Break("0")
                        elif True:
                            d_20_budgetLeft3_: int
                            d_20_budgetLeft3_ = (maxSteps) - (d_2_steps_)
                            d_21_reserveThreshold_: int
                            if (maxSteps) >= (50):
                                d_21_reserveThreshold_ = _dafny.euclidian_division(maxSteps, 10)
                            elif True:
                                d_21_reserveThreshold_ = 5
                            if (d_20_budgetLeft3_) <= (d_21_reserveThreshold_):
                                d_22_cg_: _dafny.Seq
                                d_23_ci_: bool
                                d_24_cc_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_budgetLeft3_)
                                d_22_cg_ = out13_
                                d_23_ci_ = out14_
                                d_24_cc_ = out15_
                                generated = d_22_cg_
                                insideConstrainedOut = d_23_ci_
                                currentConstrainedOut = d_24_cc_
                                d_2_steps_ = (d_2_steps_) + (d_20_budgetLeft3_)
                                if not(insideConstrainedOut):
                                    d_6_seenCompleteSpan_ = True
                                raise _dafny.Break("0")
                            elif True:
                                d_25_constrainedPrompt_: _dafny.Seq
                                d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_26_next_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_26_next_ = out16_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_26_next_) == (eosToken):
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                        d_27_cg_: _dafny.Seq
                                        d_28_ci_: bool
                                        d_29_cc_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out19_: _dafny.Seq
                                        out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_27_cg_ = out17_
                                        d_28_ci_ = out18_
                                        d_29_cc_ = out19_
                                        generated = d_27_cg_
                                        insideConstrainedOut = d_28_ci_
                                        currentConstrainedOut = d_29_cc_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        d_6_seenCompleteSpan_ = True
                                    elif (d_2_steps_) < (maxSteps):
                                        d_30_closeBudget_: int
                                        d_30_closeBudget_ = (maxSteps) - (d_2_steps_)
                                        d_31_cg_: _dafny.Seq
                                        d_32_ci_: bool
                                        d_33_cc_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudget_)
                                        d_31_cg_ = out20_
                                        d_32_ci_ = out21_
                                        d_33_cc_ = out22_
                                        generated = d_31_cg_
                                        insideConstrainedOut = d_32_ci_
                                        currentConstrainedOut = d_33_cc_
                                        d_2_steps_ = (d_2_steps_) + (d_30_closeBudget_)
                                        if not(insideConstrainedOut):
                                            d_6_seenCompleteSpan_ = True
                                    raise _dafny.Break("0")
                                elif True:
                                    d_34_ag_: _dafny.Seq
                                    d_35_ai_: bool
                                    d_36_ac_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                    d_34_ag_ = out23_
                                    d_35_ai_ = out24_
                                    d_36_ac_ = out25_
                                    generated = d_34_ag_
                                    insideConstrainedOut = d_35_ai_
                                    currentConstrainedOut = d_36_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

