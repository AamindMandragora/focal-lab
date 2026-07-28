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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the template variable names (like n1, frac1, count, etc.). Express the final answer as a Python-style mathematical expression using those template variables inside << >> delimiters. IMPORTANT: (1) Use int() to convert float results to integers, e.g., <<int(n * frac)>> or <<n - int(n * frac)>>. (2) Use // for integer floor division instead of /, e.g., <<a * b // 60>> instead of <<a * b / 60>>. (3) Do NOT use ** for exponentiation. (4) The expression must only use basic operators: +, -, *, /, //, (, ), int(), and the template variable names. Example: if n people each earn frac of their salary, total is <<int(n * frac)>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_shouldForce_: bool
                        d_6_shouldForce_ = ((not(d_5_forcedFinalSpan_)) and ((d_2_steps_) >= (d_4_freeStepsTarget_))) and (((maxSteps) - (d_2_steps_)) >= (3))
                        if d_6_shouldForce_:
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
                            d_5_forcedFinalSpan_ = True
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
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_11_cg_: _dafny.Seq
                            d_12_ci_: bool
                            d_13_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_cg_ = out7_
                            d_12_ci_ = out8_
                            d_13_cc_ = out9_
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_5_forcedFinalSpan_:
                                raise _dafny.Break("0")
                        elif ((maxSteps) - (d_2_steps_)) <= (4):
                            d_14_closeBudget_: int
                            d_14_closeBudget_ = (maxSteps) - (d_2_steps_)
                            d_15_cg_: _dafny.Seq
                            d_16_ci_: bool
                            d_17_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
                            d_15_cg_ = out10_
                            d_16_ci_ = out11_
                            d_17_cc_ = out12_
                            generated = d_15_cg_
                            insideConstrainedOut = d_16_ci_
                            currentConstrainedOut = d_17_cc_
                            d_2_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_19_next_ = out13_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                if ((maxSteps) - (d_2_steps_)) >= (2):
                                    d_20_closeBudget_: int
                                    d_20_closeBudget_ = (maxSteps) - (d_2_steps_)
                                    if (d_20_closeBudget_) > (8):
                                        d_20_closeBudget_ = 8
                                    d_21_cg_: _dafny.Seq
                                    d_22_ci_: bool
                                    d_23_cc_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudget_)
                                    d_21_cg_ = out14_
                                    d_22_ci_ = out15_
                                    d_23_cc_ = out16_
                                    generated = d_21_cg_
                                    insideConstrainedOut = d_22_ci_
                                    currentConstrainedOut = d_23_cc_
                                    d_2_steps_ = (d_2_steps_) + (d_20_closeBudget_)
                                raise _dafny.Break("0")
                            elif True:
                                d_24_ag_: _dafny.Seq
                                d_25_ai_: bool
                                d_26_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_24_ag_ = out17_
                                d_25_ai_ = out18_
                                d_26_ac_ = out19_
                                generated = d_24_ag_
                                insideConstrainedOut = d_25_ai_
                                currentConstrainedOut = d_26_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

