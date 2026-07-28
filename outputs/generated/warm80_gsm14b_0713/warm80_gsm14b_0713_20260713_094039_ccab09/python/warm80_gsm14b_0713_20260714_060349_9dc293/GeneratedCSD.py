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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the template variable names (like n1, frac1, count, etc.). Express the final answer as a Python-style mathematical expression using those template variables inside << >> delimiters. CRITICAL RULES: (1) Use int() to convert float/decimal results to integers, e.g., <<int(n * frac)>>, <<n - int(n * frac)>>, <<int(n1 * frac1 + n2 * frac2)>>. (2) Use // for integer floor division instead of /, e.g., <<(n1 - n2) * l * p * t // 60>> instead of <<(n1 - n2) * l * p * t / 60>>. (3) Do NOT use ** for exponentiation. (4) Always wrap the final answer in int() if the result should be a whole number. (5) Use only: +, -, *, /, //, (, ), int(), and template variable names."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_pendingOpenAngle_: bool
        d_6_pendingOpenAngle_ = False
        d_7_spanInterceptStep_: int
        d_7_spanInterceptStep_ = 150
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_shouldForce_: bool
                        d_8_shouldForce_ = ((not(d_5_forcedFinalSpan_)) and ((d_2_steps_) >= (d_4_freeStepsTarget_))) and (((maxSteps) - (d_2_steps_)) >= (3))
                        if d_8_shouldForce_:
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
                            d_6_pendingOpenAngle_ = False
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and ((d_2_steps_) > (d_7_spanInterceptStep_))) and (not(d_5_forcedFinalSpan_)):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                d_13_og_: _dafny.Seq
                                d_14_oi_: bool
                                d_15_oc_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_13_og_ = out4_
                                d_14_oi_ = out5_
                                d_15_oc_ = out6_
                                generated = d_13_og_
                                insideConstrainedOut = d_14_oi_
                                currentConstrainedOut = d_15_oc_
                                d_5_forcedFinalSpan_ = True
                                d_6_pendingOpenAngle_ = False
                            elif ((((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")))) and (d_6_pendingOpenAngle_)) and ((d_2_steps_) > (d_7_spanInterceptStep_))) and (not(d_5_forcedFinalSpan_)):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                d_16_og_: _dafny.Seq
                                d_17_oi_: bool
                                d_18_oc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_16_og_ = out7_
                                d_17_oi_ = out8_
                                d_18_oc_ = out9_
                                generated = d_16_og_
                                insideConstrainedOut = d_17_oi_
                                currentConstrainedOut = d_18_oc_
                                d_5_forcedFinalSpan_ = True
                                d_6_pendingOpenAngle_ = False
                            elif (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                d_6_pendingOpenAngle_ = True
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                d_6_pendingOpenAngle_ = False
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            if d_5_forcedFinalSpan_:
                                raise _dafny.Break("0")
                        elif ((maxSteps) - (d_2_steps_)) <= (4):
                            d_22_closeBudget_: int
                            d_22_closeBudget_ = (maxSteps) - (d_2_steps_)
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
                                if ((maxSteps) - (d_2_steps_)) >= (1):
                                    d_28_closeBudget_: int
                                    d_28_closeBudget_ = (maxSteps) - (d_2_steps_)
                                    if (d_28_closeBudget_) > (10):
                                        d_28_closeBudget_ = 10
                                    d_29_cg_: _dafny.Seq
                                    d_30_ci_: bool
                                    d_31_cc_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_28_closeBudget_)
                                    d_29_cg_ = out17_
                                    d_30_ci_ = out18_
                                    d_31_cc_ = out19_
                                    generated = d_29_cg_
                                    insideConstrainedOut = d_30_ci_
                                    currentConstrainedOut = d_31_cc_
                                    d_2_steps_ = (d_2_steps_) + (d_28_closeBudget_)
                                raise _dafny.Break("0")
                            elif True:
                                d_32_ag_: _dafny.Seq
                                d_33_ai_: bool
                                d_34_ac_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
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

