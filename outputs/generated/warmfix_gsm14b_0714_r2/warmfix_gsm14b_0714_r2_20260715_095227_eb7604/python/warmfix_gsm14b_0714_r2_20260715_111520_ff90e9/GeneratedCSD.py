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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Show your reasoning briefly. At the very end, write your final answer as a symbolic expression using the variable names from the problem inside << >> delimiters (e.g., <<n*(k+1)/2>> or <<x*k*(12//n)>>). Output only one final << >> span at the end."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = 75
        if (maxSteps) < (150):
            d_4_freeStepsTarget_ = _dafny.euclidian_division(maxSteps, 2)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_done_: bool
        d_6_done_ = False
        while ((d_2_steps_) < (maxSteps)) and (not(d_6_done_)):
            if not(insideConstrainedOut):
                d_7_budgetLeft_: int
                d_7_budgetLeft_ = (maxSteps) - (d_2_steps_)
                d_8_shouldForce_: bool
                d_8_shouldForce_ = (not(d_5_forcedFinalSpan_)) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or ((d_7_budgetLeft_) <= (10)))
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
                elif (d_7_budgetLeft_) == (0):
                    d_6_done_ = True
                elif True:
                    d_12_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_12_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_12_next_) == (eosToken):
                        if (not(d_5_forcedFinalSpan_)) and (((maxSteps) - (d_2_steps_)) >= (2)):
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
                            d_6_done_ = True
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
            elif True:
                d_16_budgetLeft2_: int
                d_16_budgetLeft2_ = (maxSteps) - (d_2_steps_)
                if (parser).IsCompletePrefix(currentConstrainedOut):
                    d_17_cg_: _dafny.Seq
                    d_18_ci_: bool
                    d_19_cc_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_17_cg_ = out7_
                    d_18_ci_ = out8_
                    d_19_cc_ = out9_
                    generated = d_17_cg_
                    insideConstrainedOut = d_18_ci_
                    currentConstrainedOut = d_19_cc_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_6_done_ = True
                elif (d_16_budgetLeft2_) <= (5):
                    d_20_closeBudget_: int
                    d_20_closeBudget_ = d_16_budgetLeft2_
                    d_21_cg_: _dafny.Seq
                    d_22_ci_: bool
                    d_23_cc_: _dafny.Seq
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudget_)
                    d_21_cg_ = out10_
                    d_22_ci_ = out11_
                    d_23_cc_ = out12_
                    generated = d_21_cg_
                    insideConstrainedOut = d_22_ci_
                    currentConstrainedOut = d_23_cc_
                    d_2_steps_ = (d_2_steps_) + (d_20_closeBudget_)
                    d_6_done_ = True
                elif True:
                    d_24_constrainedPrompt_: _dafny.Seq
                    d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_25_next_: _dafny.Seq
                    out13_: _dafny.Seq
                    out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                    d_25_next_ = out13_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_25_next_) == (eosToken):
                        d_26_budgetLeft3_: int
                        d_26_budgetLeft3_ = (maxSteps) - (d_2_steps_)
                        if (d_26_budgetLeft3_) >= (1):
                            d_27_closeBudget2_: int
                            d_27_closeBudget2_ = d_26_budgetLeft3_
                            d_28_cg2_: _dafny.Seq
                            d_29_ci2_: bool
                            d_30_cc2_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget2_)
                            d_28_cg2_ = out14_
                            d_29_ci2_ = out15_
                            d_30_cc2_ = out16_
                            generated = d_28_cg2_
                            insideConstrainedOut = d_29_ci2_
                            currentConstrainedOut = d_30_cc2_
                            d_2_steps_ = (d_2_steps_) + (d_27_closeBudget2_)
                        d_6_done_ = True
                    elif True:
                        d_31_ag_: _dafny.Seq
                        d_32_ai_: bool
                        d_33_ac_: _dafny.Seq
                        out17_: _dafny.Seq
                        out18_: bool
                        out19_: _dafny.Seq
                        out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                        d_31_ag_ = out17_
                        d_32_ai_ = out18_
                        d_33_ac_ = out19_
                        generated = d_31_ag_
                        insideConstrainedOut = d_32_ai_
                        currentConstrainedOut = d_33_ac_
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

