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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the variable names provided. Show your reasoning, then end with: The final answer is <<EXPR>> where EXPR uses the given variable names and arithmetic operators +, -, *, /, (, ), ^ only. Output nothing after the final <<EXPR>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        if (maxSteps) >= (10):
            d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (4), 5)
        elif True:
            d_4_freeStepsTarget_ = _dafny.euclidian_division(maxSteps, 2)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_seenCompleteSpan_: bool
        d_6_seenCompleteSpan_ = False
        d_7_postSpanFreeSteps_: int
        d_7_postSpanFreeSteps_ = 0
        d_8_postSpanBudget_: int
        d_8_postSpanBudget_ = 3
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_9_budgetLeft_: int
                        d_9_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        if (d_6_seenCompleteSpan_) and ((d_7_postSpanFreeSteps_) >= (d_8_postSpanBudget_)):
                            raise _dafny.Break("0")
                        d_10_shouldForce_: bool
                        d_10_shouldForce_ = ((not(d_5_forcedFinalSpan_)) and (not(d_6_seenCompleteSpan_))) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or ((d_9_budgetLeft_) <= (10)))
                        if (d_10_shouldForce_) and ((d_9_budgetLeft_) >= (3)):
                            d_11_og_: _dafny.Seq
                            d_12_oi_: bool
                            d_13_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_11_og_ = out0_
                            d_12_oi_ = out1_
                            d_13_oc_ = out2_
                            generated = d_11_og_
                            insideConstrainedOut = d_12_oi_
                            currentConstrainedOut = d_13_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_5_forcedFinalSpan_ = True
                            d_7_postSpanFreeSteps_ = 0
                        elif (d_10_shouldForce_) and ((d_9_budgetLeft_) < (3)):
                            raise _dafny.Break("0")
                        elif True:
                            d_14_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                                if (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    if d_6_seenCompleteSpan_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        out4_: _dafny.Seq
                                        out5_: bool
                                        out6_: _dafny.Seq
                                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                        generated = out4_
                                        insideConstrainedOut = out5_
                                        currentConstrainedOut = out6_
                                        d_7_postSpanFreeSteps_ = 0
                                elif d_6_seenCompleteSpan_:
                                    d_7_postSpanFreeSteps_ = (d_7_postSpanFreeSteps_) + (1)
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_15_cg_: _dafny.Seq
                            d_16_ci_: bool
                            d_17_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_15_cg_ = out7_
                            d_16_ci_ = out8_
                            d_17_cc_ = out9_
                            generated = d_15_cg_
                            insideConstrainedOut = d_16_ci_
                            currentConstrainedOut = d_17_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_6_seenCompleteSpan_ = True
                            d_7_postSpanFreeSteps_ = 0
                            if d_5_forcedFinalSpan_:
                                raise _dafny.Break("0")
                        elif True:
                            d_18_budgetLeft2_: int
                            d_18_budgetLeft2_ = (maxSteps) - (d_2_steps_)
                            d_19_reserveThreshold_: int
                            if (maxSteps) >= (30):
                                d_19_reserveThreshold_ = _dafny.euclidian_division(maxSteps, 15)
                            elif True:
                                d_19_reserveThreshold_ = 2
                            if (d_18_budgetLeft2_) <= (d_19_reserveThreshold_):
                                d_20_cg_: _dafny.Seq
                                d_21_ci_: bool
                                d_22_cc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_budgetLeft2_)
                                d_20_cg_ = out10_
                                d_21_ci_ = out11_
                                d_22_cc_ = out12_
                                generated = d_20_cg_
                                insideConstrainedOut = d_21_ci_
                                currentConstrainedOut = d_22_cc_
                                d_2_steps_ = maxSteps
                                if not(insideConstrainedOut):
                                    d_6_seenCompleteSpan_ = True
                                raise _dafny.Break("0")
                            elif True:
                                d_23_constrainedPrompt_: _dafny.Seq
                                d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_24_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_24_next_ = out13_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_24_next_) == (eosToken):
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                        d_25_cg_: _dafny.Seq
                                        d_26_ci_: bool
                                        d_27_cc_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_25_cg_ = out14_
                                        d_26_ci_ = out15_
                                        d_27_cc_ = out16_
                                        generated = d_25_cg_
                                        insideConstrainedOut = d_26_ci_
                                        currentConstrainedOut = d_27_cc_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        d_6_seenCompleteSpan_ = True
                                    elif (d_2_steps_) < (maxSteps):
                                        d_28_closeBudget2_: int
                                        d_28_closeBudget2_ = (maxSteps) - (d_2_steps_)
                                        d_29_cg_: _dafny.Seq
                                        d_30_ci_: bool
                                        d_31_cc_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out19_: _dafny.Seq
                                        out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_closeBudget2_)
                                        d_29_cg_ = out17_
                                        d_30_ci_ = out18_
                                        d_31_cc_ = out19_
                                        generated = d_29_cg_
                                        insideConstrainedOut = d_30_ci_
                                        currentConstrainedOut = d_31_cc_
                                        d_2_steps_ = maxSteps
                                        if not(insideConstrainedOut):
                                            d_6_seenCompleteSpan_ = True
                                    raise _dafny.Break("0")
                                elif True:
                                    d_32_ag_: _dafny.Seq
                                    d_33_ai_: bool
                                    d_34_ac_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                    d_32_ag_ = out20_
                                    d_33_ai_ = out21_
                                    d_34_ac_ = out22_
                                    generated = d_32_ag_
                                    insideConstrainedOut = d_33_ai_
                                    currentConstrainedOut = d_34_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

