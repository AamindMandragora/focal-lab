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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the symbolic variable names given. Write the final answer as a single arithmetic expression inside << >> delimiters. Use only +, -, *, /, //, % operators. Do not use ** for powers; write multiplication instead. Do not wrap in int(). Example: <<n * rate>> or <<total - n1 - n2>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (72), 100)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_spanSteps_: int
        d_6_spanSteps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_budgetLeft_: int
                        d_7_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        d_8_shouldForce_: bool
                        d_8_shouldForce_ = (not(d_5_forcedFinalSpan_)) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or ((d_7_budgetLeft_) <= (8)))
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
                            d_6_spanSteps_ = 0
                        elif (d_7_budgetLeft_) <= (1):
                            raise _dafny.Break("0")
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                if (not(d_5_forcedFinalSpan_)) and (((maxSteps) - (d_2_steps_)) >= (2)):
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
                                    d_6_spanSteps_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                    elif True:
                        d_16_budgetLeft2_: int
                        d_16_budgetLeft2_ = (maxSteps) - (d_2_steps_)
                        d_6_spanSteps_ = (d_6_spanSteps_) + (1)
                        d_17_shouldClose_: bool
                        d_17_shouldClose_ = ((d_16_budgetLeft2_) <= (5)) or ((d_6_spanSteps_) >= (20))
                        if d_17_shouldClose_:
                            if (d_16_budgetLeft2_) <= (3):
                                d_18_closeBudget_: int
                                d_18_closeBudget_ = d_16_budgetLeft2_
                                d_19_cg_: _dafny.Seq
                                d_20_ci_: bool
                                d_21_cc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
                                d_19_cg_ = out7_
                                d_20_ci_ = out8_
                                d_21_cc_ = out9_
                                generated = d_19_cg_
                                insideConstrainedOut = d_20_ci_
                                currentConstrainedOut = d_21_cc_
                                d_2_steps_ = maxSteps
                                raise _dafny.Break("0")
                            elif True:
                                d_22_cg_: _dafny.Seq
                                d_23_ci_: bool
                                d_24_cc_: _dafny.Seq
                                d_25_closed_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_22_cg_ = out10_
                                d_23_ci_ = out11_
                                d_24_cc_ = out12_
                                d_25_closed_ = out13_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if d_25_closed_:
                                    generated = d_22_cg_
                                    insideConstrainedOut = d_23_ci_
                                    currentConstrainedOut = d_24_cc_
                                    raise _dafny.Break("0")
                                elif True:
                                    if ((maxSteps) - (d_2_steps_)) >= (1):
                                        d_26_constrainedPrompt_: _dafny.Seq
                                        d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_27_next_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                        d_27_next_ = out14_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        if (d_27_next_) == (eosToken):
                                            d_28_closeBudget2_: int
                                            d_28_closeBudget2_ = (maxSteps) - (d_2_steps_)
                                            if (d_28_closeBudget2_) >= (1):
                                                d_29_cg2_: _dafny.Seq
                                                d_30_ci2_: bool
                                                d_31_cc2_: _dafny.Seq
                                                out15_: _dafny.Seq
                                                out16_: bool
                                                out17_: _dafny.Seq
                                                out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_closeBudget2_)
                                                d_29_cg2_ = out15_
                                                d_30_ci2_ = out16_
                                                d_31_cc2_ = out17_
                                                generated = d_29_cg2_
                                                insideConstrainedOut = d_30_ci2_
                                                currentConstrainedOut = d_31_cc2_
                                                d_2_steps_ = maxSteps
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_32_ag_: _dafny.Seq
                                            d_33_ai_: bool
                                            d_34_ac_: _dafny.Seq
                                            out18_: _dafny.Seq
                                            out19_: bool
                                            out20_: _dafny.Seq
                                            out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                            d_32_ag_ = out18_
                                            d_33_ai_ = out19_
                                            d_34_ac_ = out20_
                                            generated = d_32_ag_
                                            insideConstrainedOut = d_33_ai_
                                            currentConstrainedOut = d_34_ac_
                        elif True:
                            d_35_constrainedPrompt2_: _dafny.Seq
                            d_35_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_36_next2_: _dafny.Seq
                            out21_: _dafny.Seq
                            out21_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_35_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_36_next2_ = out21_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_36_next2_) == (eosToken):
                                d_37_closeBudget3_: int
                                d_37_closeBudget3_ = (maxSteps) - (d_2_steps_)
                                if (d_37_closeBudget3_) >= (1):
                                    d_38_cg3_: _dafny.Seq
                                    d_39_ci3_: bool
                                    d_40_cc3_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: bool
                                    out24_: _dafny.Seq
                                    out22_, out23_, out24_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_37_closeBudget3_)
                                    d_38_cg3_ = out22_
                                    d_39_ci3_ = out23_
                                    d_40_cc3_ = out24_
                                    generated = d_38_cg3_
                                    insideConstrainedOut = d_39_ci3_
                                    currentConstrainedOut = d_40_cc3_
                                    d_2_steps_ = maxSteps
                                raise _dafny.Break("0")
                            elif True:
                                d_41_ag2_: _dafny.Seq
                                d_42_ai2_: bool
                                d_43_ac2_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_36_next2_)
                                d_41_ag2_ = out25_
                                d_42_ai2_ = out26_
                                d_43_ac2_ = out27_
                                generated = d_41_ag2_
                                insideConstrainedOut = d_42_ai2_
                                currentConstrainedOut = d_43_ac2_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

