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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final answer as a formula expression inside << >> at the very end, using variable names without curly braces. Example: <<n * (rate + 1)>> or <<total - n1 - n2>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_lastSpanLength_: int
        d_4_lastSpanLength_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_budgetUsed_: int
                        d_5_budgetUsed_ = d_2_steps_
                        d_6_budgetLeft_: int
                        d_6_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        if ((d_6_budgetLeft_) <= (5)) and ((d_6_budgetLeft_) >= (1)):
                            if (d_6_budgetLeft_) >= (2):
                                d_7_og_: _dafny.Seq
                                d_8_oi_: bool
                                d_9_oc_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_7_og_ = out0_
                                d_8_oi_ = out1_
                                d_9_oc_ = out2_
                                generated = d_7_og_
                                insideConstrainedOut = d_8_oi_
                                currentConstrainedOut = d_9_oc_
                                d_2_steps_ = (d_2_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out4_
                                    insideConstrainedOut = out5_
                                    currentConstrainedOut = out6_
                    elif True:
                        d_11_budgetLeft2_: int
                        d_11_budgetLeft2_ = (maxSteps) - (d_2_steps_)
                        if (d_11_budgetLeft2_) <= (3):
                            d_12_closeBudget_: int
                            d_12_closeBudget_ = d_11_budgetLeft2_
                            d_13_cg_: _dafny.Seq
                            d_14_ci_: bool
                            d_15_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget_)
                            d_13_cg_ = out7_
                            d_14_ci_ = out8_
                            d_15_cc_ = out9_
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                            d_2_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            d_16_cg_: _dafny.Seq
                            d_17_ci_: bool
                            d_18_cc_: _dafny.Seq
                            d_19_closed_: bool
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out13_: bool
                            out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_16_cg_ = out10_
                            d_17_ci_ = out11_
                            d_18_cc_ = out12_
                            d_19_closed_ = out13_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_19_closed_:
                                d_4_lastSpanLength_ = len(currentConstrainedOut)
                                generated = d_16_cg_
                                insideConstrainedOut = d_17_ci_
                                currentConstrainedOut = d_18_cc_
                                if ((d_4_lastSpanLength_) >= (3)) and (((d_2_steps_) * (2)) >= (maxSteps)):
                                    raise _dafny.Break("0")
                                if ((maxSteps) - (d_2_steps_)) <= (10):
                                    raise _dafny.Break("0")
                            elif True:
                                d_20_isDeadEnd_: bool
                                out14_: bool
                                out14_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                                d_20_isDeadEnd_ = out14_
                                if d_20_isDeadEnd_:
                                    d_21_closeBudget3_: int
                                    d_21_closeBudget3_ = (maxSteps) - (d_2_steps_)
                                    d_22_cg3_: _dafny.Seq
                                    d_23_ci3_: bool
                                    d_24_cc3_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget3_)
                                    d_22_cg3_ = out15_
                                    d_23_ci3_ = out16_
                                    d_24_cc3_ = out17_
                                    generated = d_22_cg3_
                                    insideConstrainedOut = d_23_ci3_
                                    currentConstrainedOut = d_24_cc3_
                                    d_2_steps_ = maxSteps
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_constrainedPrompt_: _dafny.Seq
                                    d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_26_next_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                    d_26_next_ = out18_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    if (d_26_next_) == (eosToken):
                                        if (d_2_steps_) < (maxSteps):
                                            d_27_closeBudget4_: int
                                            d_27_closeBudget4_ = (maxSteps) - (d_2_steps_)
                                            if (d_27_closeBudget4_) >= (1):
                                                d_28_cg4_: _dafny.Seq
                                                d_29_ci4_: bool
                                                d_30_cc4_: _dafny.Seq
                                                out19_: _dafny.Seq
                                                out20_: bool
                                                out21_: _dafny.Seq
                                                out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget4_)
                                                d_28_cg4_ = out19_
                                                d_29_ci4_ = out20_
                                                d_30_cc4_ = out21_
                                                generated = d_28_cg4_
                                                insideConstrainedOut = d_29_ci4_
                                                currentConstrainedOut = d_30_cc4_
                                                d_2_steps_ = maxSteps
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_31_ag_: _dafny.Seq
                                        d_32_ai_: bool
                                        d_33_ac_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                        d_31_ag_ = out22_
                                        d_32_ai_ = out23_
                                        d_33_ac_ = out24_
                                        generated = d_31_ag_
                                        insideConstrainedOut = d_32_ai_
                                        currentConstrainedOut = d_33_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

