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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step using ONLY actual numeric values (e.g. 3, 4.5, 100). Never use variable names, braces, or symbolic placeholders inside << >>. Every << >> must contain only digits, +, -, *, /, (, ) and decimal points. Final answer must be a number. Example: <<3*4+2>> or <<42>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeReasoningBudget_: int
        d_4_freeReasoningBudget_ = _dafny.euclidian_division((maxSteps) * (55), 100)
        d_5_spanReserve_: int
        d_5_spanReserve_ = 8
        with _dafny.label("0"):
            while ((d_2_steps_) < (d_4_freeReasoningBudget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_6_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_6_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_6_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out1_
                            insideConstrainedOut = out2_
                            currentConstrainedOut = out3_
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_7_og_: _dafny.Seq
            d_8_oi_: bool
            d_9_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_7_og_ = out4_
            d_8_oi_ = out5_
            d_9_oc_ = out6_
            generated = d_7_og_
            insideConstrainedOut = d_8_oi_
            currentConstrainedOut = d_9_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("1"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("1"):
                    d_10_remainingSteps_: int
                    d_10_remainingSteps_ = (maxSteps) - (d_2_steps_)
                    if not(insideConstrainedOut):
                        if (d_10_remainingSteps_) <= (d_5_spanReserve_):
                            if (d_2_steps_) < (maxSteps):
                                d_11_og_: _dafny.Seq
                                d_12_oi_: bool
                                d_13_oc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_11_og_ = out7_
                                d_12_oi_ = out8_
                                d_13_oc_ = out9_
                                generated = d_11_og_
                                insideConstrainedOut = d_12_oi_
                                currentConstrainedOut = d_13_oc_
                                d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_14_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_next_ = out10_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                                if (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out11_
                                    insideConstrainedOut = out12_
                                    currentConstrainedOut = out13_
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_15_cg_: _dafny.Seq
                            d_16_ci_: bool
                            d_17_cc_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_15_cg_ = out14_
                            d_16_ci_ = out15_
                            d_17_cc_ = out16_
                            generated = d_15_cg_
                            insideConstrainedOut = d_16_ci_
                            currentConstrainedOut = d_17_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif (d_10_remainingSteps_) <= (d_5_spanReserve_):
                            d_18_closeBudget_: int
                            d_18_closeBudget_ = (maxSteps) - (d_2_steps_)
                            d_19_cg_: _dafny.Seq
                            d_20_ci_: bool
                            d_21_cc_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
                            d_19_cg_ = out17_
                            d_20_ci_ = out18_
                            d_21_cc_ = out19_
                            generated = d_19_cg_
                            insideConstrainedOut = d_20_ci_
                            currentConstrainedOut = d_21_cc_
                            d_2_steps_ = maxSteps
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out20_: _dafny.Seq
                            out20_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_23_next_ = out20_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_23_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_24_ag_: _dafny.Seq
                                d_25_ai_: bool
                                d_26_ac_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_24_ag_ = out21_
                                d_25_ai_ = out22_
                                d_26_ac_ = out23_
                                generated = d_24_ag_
                                insideConstrainedOut = d_25_ai_
                                currentConstrainedOut = d_26_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

