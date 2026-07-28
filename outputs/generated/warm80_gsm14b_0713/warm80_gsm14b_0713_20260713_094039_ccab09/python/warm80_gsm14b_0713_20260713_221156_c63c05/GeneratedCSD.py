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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. The problem contains specific numeric values. Compute with those actual numbers. At the end, output the final numeric answer inside << >> delimiters (e.g. <<42>>). The final answer must be a plain integer or decimal number."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_lastClosedSpanLength_: int
        d_6_lastClosedSpanLength_ = 0
        d_7_anySpanClosed_: bool
        d_7_anySpanClosed_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_budgetLeft_: int
                        d_8_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        d_9_shouldForce_: bool
                        d_9_shouldForce_ = (not(d_5_forcedFinalSpan_)) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or ((d_8_budgetLeft_) <= (5)))
                        if (d_9_shouldForce_) and ((d_8_budgetLeft_) >= (2)):
                            d_10_og_: _dafny.Seq
                            d_11_oi_: bool
                            d_12_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_og_ = out0_
                            d_11_oi_ = out1_
                            d_12_oc_ = out2_
                            generated = d_10_og_
                            insideConstrainedOut = d_11_oi_
                            currentConstrainedOut = d_12_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_5_forcedFinalSpan_ = True
                        elif d_9_shouldForce_:
                            raise _dafny.Break("0")
                        elif True:
                            d_13_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                if (not(d_5_forcedFinalSpan_)) and ((d_8_budgetLeft_) >= (2)):
                                    d_14_og_: _dafny.Seq
                                    d_15_oi_: bool
                                    d_16_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_14_og_ = out4_
                                    d_15_oi_ = out5_
                                    d_16_oc_ = out6_
                                    generated = d_14_og_
                                    insideConstrainedOut = d_15_oi_
                                    currentConstrainedOut = d_16_oc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_5_forcedFinalSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                if (((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (d_7_anySpanClosed_)) and ((d_6_lastClosedSpanLength_) >= (3)):
                                    raise _dafny.Break("0")
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out7_
                                    insideConstrainedOut = out8_
                                    currentConstrainedOut = out9_
                    elif True:
                        d_17_budgetLeft2_: int
                        d_17_budgetLeft2_ = (maxSteps) - (d_2_steps_)
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            if (d_17_budgetLeft2_) >= (1):
                                d_18_closedSpanLength_: int
                                d_18_closedSpanLength_ = len(currentConstrainedOut)
                                d_19_cg_: _dafny.Seq
                                d_20_ci_: bool
                                d_21_cc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_19_cg_ = out10_
                                d_20_ci_ = out11_
                                d_21_cc_ = out12_
                                generated = d_19_cg_
                                insideConstrainedOut = d_20_ci_
                                currentConstrainedOut = d_21_cc_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_7_anySpanClosed_ = True
                                d_6_lastClosedSpanLength_ = d_18_closedSpanLength_
                                if d_5_forcedFinalSpan_:
                                    raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif (d_17_budgetLeft2_) <= (4):
                            d_22_closeBudget_: int
                            d_22_closeBudget_ = d_17_budgetLeft2_
                            d_23_cg_: _dafny.Seq
                            d_24_ci_: bool
                            d_25_cc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
                            d_23_cg_ = out13_
                            d_24_ci_ = out14_
                            d_25_cc_ = out15_
                            generated = d_23_cg_
                            insideConstrainedOut = d_24_ci_
                            currentConstrainedOut = d_25_cc_
                            d_2_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            d_26_constrainedPrompt_: _dafny.Seq
                            d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_27_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_27_next_ = out16_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_27_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_28_closedSpanLength_: int
                                    d_28_closedSpanLength_ = len(currentConstrainedOut)
                                    d_29_cg_: _dafny.Seq
                                    d_30_ci_: bool
                                    d_31_cc_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_29_cg_ = out17_
                                    d_30_ci_ = out18_
                                    d_31_cc_ = out19_
                                    generated = d_29_cg_
                                    insideConstrainedOut = d_30_ci_
                                    currentConstrainedOut = d_31_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_7_anySpanClosed_ = True
                                    d_6_lastClosedSpanLength_ = d_28_closedSpanLength_
                                elif (d_2_steps_) < (maxSteps):
                                    d_32_closeBudget2_: int
                                    d_32_closeBudget2_ = (maxSteps) - (d_2_steps_)
                                    d_33_cg_: _dafny.Seq
                                    d_34_ci_: bool
                                    d_35_cc_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudget2_)
                                    d_33_cg_ = out20_
                                    d_34_ci_ = out21_
                                    d_35_cc_ = out22_
                                    generated = d_33_cg_
                                    insideConstrainedOut = d_34_ci_
                                    currentConstrainedOut = d_35_cc_
                                    d_2_steps_ = maxSteps
                                raise _dafny.Break("0")
                            elif True:
                                d_36_ag_: _dafny.Seq
                                d_37_ai_: bool
                                d_38_ac_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                d_36_ag_ = out23_
                                d_37_ai_ = out24_
                                d_38_ac_ = out25_
                                generated = d_36_ag_
                                insideConstrainedOut = d_37_ai_
                                currentConstrainedOut = d_38_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

