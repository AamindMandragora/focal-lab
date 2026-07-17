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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using only the variable names given. Show your work. At the very end write: The final answer is <<EXPR>> where EXPR uses only the exact variable names provided and operators +, -, *, /, (, ) with no spaces inside the expression. Do NOT use {}, int(), //, **, or any other Python-specific syntax inside << >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        if (maxSteps) >= (10):
            d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (85), 100)
        elif True:
            d_4_freeStepsTarget_ = _dafny.euclidian_division(maxSteps, 2)
        d_5_forcedSpanOpened_: bool
        d_5_forcedSpanOpened_ = False
        d_6_spanDone_: bool
        d_6_spanDone_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_6_spanDone_:
                            raise _dafny.Break("0")
                        d_7_budgetLeft_: int
                        d_7_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        d_8_shouldForce_: bool
                        d_8_shouldForce_ = (not(d_5_forcedSpanOpened_)) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or ((d_7_budgetLeft_) <= (8)))
                        if (d_8_shouldForce_) and ((d_7_budgetLeft_) >= (3)):
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
                            d_5_forcedSpanOpened_ = True
                        elif (d_8_shouldForce_) and ((d_7_budgetLeft_) < (3)):
                            raise _dafny.Break("0")
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                if (not(d_5_forcedSpanOpened_)) and ((d_2_steps_) < (maxSteps)):
                                    pass
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(d_5_forcedSpanOpened_)):
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out4_
                                    insideConstrainedOut = out5_
                                    currentConstrainedOut = out6_
                                    d_5_forcedSpanOpened_ = True
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_13_budgetLeft2_: int
                            d_13_budgetLeft2_ = (maxSteps) - (d_2_steps_)
                            if (d_13_budgetLeft2_) >= (1):
                                d_14_cg_: _dafny.Seq
                                d_15_ci_: bool
                                d_16_cc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_14_cg_ = out7_
                                d_15_ci_ = out8_
                                d_16_cc_ = out9_
                                generated = d_14_cg_
                                insideConstrainedOut = d_15_ci_
                                currentConstrainedOut = d_16_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_6_spanDone_ = True
                                raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_17_budgetLeft3_: int
                            d_17_budgetLeft3_ = (maxSteps) - (d_2_steps_)
                            d_18_closeReserve_: int
                            d_18_closeReserve_ = 3
                            if (d_17_budgetLeft3_) <= (d_18_closeReserve_):
                                d_19_cg_: _dafny.Seq
                                d_20_ci_: bool
                                d_21_cc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_budgetLeft3_)
                                d_19_cg_ = out10_
                                d_20_ci_ = out11_
                                d_21_cc_ = out12_
                                generated = d_19_cg_
                                insideConstrainedOut = d_20_ci_
                                currentConstrainedOut = d_21_cc_
                                d_2_steps_ = maxSteps
                                d_6_spanDone_ = True
                                raise _dafny.Break("0")
                            elif True:
                                d_22_constrainedPrompt_: _dafny.Seq
                                d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_23_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_23_next_ = out13_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_23_next_) == (eosToken):
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                        d_24_cg_: _dafny.Seq
                                        d_25_ci_: bool
                                        d_26_cc_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_24_cg_ = out14_
                                        d_25_ci_ = out15_
                                        d_26_cc_ = out16_
                                        generated = d_24_cg_
                                        insideConstrainedOut = d_25_ci_
                                        currentConstrainedOut = d_26_cc_
                                        d_2_steps_ = (d_2_steps_) + (1)
                                        d_6_spanDone_ = True
                                    elif (d_2_steps_) < (maxSteps):
                                        d_27_closeBudget_: int
                                        d_27_closeBudget_ = (maxSteps) - (d_2_steps_)
                                        d_28_cg_: _dafny.Seq
                                        d_29_ci_: bool
                                        d_30_cc_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out19_: _dafny.Seq
                                        out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget_)
                                        d_28_cg_ = out17_
                                        d_29_ci_ = out18_
                                        d_30_cc_ = out19_
                                        generated = d_28_cg_
                                        insideConstrainedOut = d_29_ci_
                                        currentConstrainedOut = d_30_cc_
                                        d_2_steps_ = maxSteps
                                        d_6_spanDone_ = True
                                    raise _dafny.Break("0")
                                elif True:
                                    d_31_ag_: _dafny.Seq
                                    d_32_ai_: bool
                                    d_33_ac_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                    d_31_ag_ = out20_
                                    d_32_ai_ = out21_
                                    d_33_ac_ = out22_
                                    generated = d_31_ag_
                                    insideConstrainedOut = d_32_ai_
                                    currentConstrainedOut = d_33_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

