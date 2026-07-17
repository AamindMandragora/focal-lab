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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using symbolic variable names. At the very end, write ONLY the final answer expression inside << >> (e.g. <<n * rate>> or <<a + b - c>>). Use only +, -, *, /, // operators. Do not use ** or {}. Write one single final << >> with the answer expression."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_hasClosedASpan_: bool
        d_6_hasClosedASpan_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_budgetLeft_: int
                        d_7_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        d_8_shouldForce_: bool
                        d_8_shouldForce_ = ((not(d_5_forcedFinalSpan_)) and (not(d_6_hasClosedASpan_))) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or ((d_7_budgetLeft_) <= (5)))
                        if (d_8_shouldForce_) and ((d_7_budgetLeft_) >= (2)):
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
                        elif (d_7_budgetLeft_) <= (1):
                            raise _dafny.Break("0")
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                if ((not(d_6_hasClosedASpan_)) and (not(d_5_forcedFinalSpan_))) and (((maxSteps) - (d_2_steps_)) >= (2)):
                                    d_13_og2_: _dafny.Seq
                                    d_14_oi2_: bool
                                    d_15_oc2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_13_og2_ = out4_
                                    d_14_oi2_ = out5_
                                    d_15_oc2_ = out6_
                                    generated = d_13_og2_
                                    insideConstrainedOut = d_14_oi2_
                                    currentConstrainedOut = d_15_oc2_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_5_forcedFinalSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out7_
                                    insideConstrainedOut = out8_
                                    currentConstrainedOut = out9_
                    elif True:
                        d_16_budgetLeft2_: int
                        d_16_budgetLeft2_ = (maxSteps) - (d_2_steps_)
                        if (d_16_budgetLeft2_) <= (4):
                            d_17_closeBudget_: int
                            d_17_closeBudget_ = d_16_budgetLeft2_
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                            d_18_cg_ = out10_
                            d_19_ci_ = out11_
                            d_20_cc_ = out12_
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_2_steps_ = maxSteps
                            d_6_hasClosedASpan_ = True
                            raise _dafny.Break("0")
                        elif True:
                            d_21_cg_: _dafny.Seq
                            d_22_ci_: bool
                            d_23_cc_: _dafny.Seq
                            d_24_closed_: bool
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out13_, out14_, out15_, out16_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_21_cg_ = out13_
                            d_22_ci_ = out14_
                            d_23_cc_ = out15_
                            d_24_closed_ = out16_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_24_closed_:
                                generated = d_21_cg_
                                insideConstrainedOut = d_22_ci_
                                currentConstrainedOut = d_23_cc_
                                d_6_hasClosedASpan_ = True
                                if d_5_forcedFinalSpan_:
                                    raise _dafny.Break("0")
                                if ((maxSteps) - (d_2_steps_)) <= (8):
                                    raise _dafny.Break("0")
                            elif True:
                                if ((maxSteps) - (d_2_steps_)) >= (1):
                                    d_25_constrainedPrompt_: _dafny.Seq
                                    d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_26_next2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                    d_26_next2_ = out17_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    if (d_26_next2_) == (eosToken):
                                        d_27_closeBudget2_: int
                                        d_27_closeBudget2_ = (maxSteps) - (d_2_steps_)
                                        if (d_27_closeBudget2_) >= (1):
                                            d_28_cg2_: _dafny.Seq
                                            d_29_ci2_: bool
                                            d_30_cc2_: _dafny.Seq
                                            out18_: _dafny.Seq
                                            out19_: bool
                                            out20_: _dafny.Seq
                                            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget2_)
                                            d_28_cg2_ = out18_
                                            d_29_ci2_ = out19_
                                            d_30_cc2_ = out20_
                                            generated = d_28_cg2_
                                            insideConstrainedOut = d_29_ci2_
                                            currentConstrainedOut = d_30_cc2_
                                            d_2_steps_ = maxSteps
                                            d_6_hasClosedASpan_ = True
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_31_ag_: _dafny.Seq
                                        d_32_ai_: bool
                                        d_33_ac_: _dafny.Seq
                                        out21_: _dafny.Seq
                                        out22_: bool
                                        out23_: _dafny.Seq
                                        out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next2_)
                                        d_31_ag_ = out21_
                                        d_32_ai_ = out22_
                                        d_33_ac_ = out23_
                                        generated = d_31_ag_
                                        insideConstrainedOut = d_32_ai_
                                        currentConstrainedOut = d_33_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

