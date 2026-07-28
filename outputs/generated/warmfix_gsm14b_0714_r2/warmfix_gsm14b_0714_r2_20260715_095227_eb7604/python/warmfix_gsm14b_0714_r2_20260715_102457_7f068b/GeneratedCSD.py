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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the given variable names. Show each calculation. At the end, write 'The final answer is' and put the complete symbolic expression (using the variable names from the problem) inside << >> delimiters."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_reserveBudget_: int
        d_4_reserveBudget_ = 60
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and ((((maxSteps) - (d_2_steps_)) > (d_4_reserveBudget_)) or ((d_2_steps_) == (0))):
                with _dafny.c_label("0"):
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                d_6_eg_: _dafny.Seq
                d_7_ei_: bool
                d_8_ec_: _dafny.Seq
                out1_: _dafny.Seq
                out2_: bool
                out3_: _dafny.Seq
                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_6_eg_ = out1_
                d_7_ei_ = out2_
                d_8_ec_ = out3_
                generated = d_6_eg_
                insideConstrainedOut = d_7_ei_
                currentConstrainedOut = d_8_ec_
                d_9_closeBudget_: int
                d_9_closeBudget_ = (maxSteps) - (d_2_steps_)
                if (d_9_closeBudget_) > (0):
                    d_10_cg_: _dafny.Seq
                    d_11_ci_: bool
                    d_12_cc_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_closeBudget_)
                    d_10_cg_ = out4_
                    d_11_ci_ = out5_
                    d_12_cc_ = out6_
                    generated = d_10_cg_
                    insideConstrainedOut = d_11_ci_
                    currentConstrainedOut = d_12_cc_
                    d_2_steps_ = (d_2_steps_) + (d_9_closeBudget_)
            elif True:
                d_13_og_: _dafny.Seq
                d_14_oi_: bool
                d_15_oc_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_13_og_ = out7_
                d_14_oi_ = out8_
                d_15_oc_ = out9_
                generated = d_13_og_
                insideConstrainedOut = d_14_oi_
                currentConstrainedOut = d_15_oc_
                d_2_steps_ = (d_2_steps_) + (1)
                with _dafny.label("1_1_0"):
                    while (d_2_steps_) < (maxSteps):
                        with _dafny.c_label("1_1_0"):
                            if not(insideConstrainedOut):
                                raise _dafny.Break("1_1_0")
                            if ((maxSteps) - (d_2_steps_)) <= (5):
                                d_16_closeBudget_: int
                                d_16_closeBudget_ = (maxSteps) - (d_2_steps_)
                                d_17_cg_: _dafny.Seq
                                d_18_ci_: bool
                                d_19_cc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
                                d_17_cg_ = out10_
                                d_18_ci_ = out11_
                                d_19_cc_ = out12_
                                generated = d_17_cg_
                                insideConstrainedOut = d_18_ci_
                                currentConstrainedOut = d_19_cc_
                                d_2_steps_ = (d_2_steps_) + (d_16_closeBudget_)
                                raise _dafny.Break("1_1_0")
                            d_20_cg_: _dafny.Seq
                            d_21_ci_: bool
                            d_22_cc_: _dafny.Seq
                            d_23_closed_: bool
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out13_, out14_, out15_, out16_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_20_cg_ = out13_
                            d_21_ci_ = out14_
                            d_22_cc_ = out15_
                            d_23_closed_ = out16_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_23_closed_:
                                generated = d_20_cg_
                                insideConstrainedOut = d_21_ci_
                                currentConstrainedOut = d_22_cc_
                                raise _dafny.Break("1_1_0")
                            elif True:
                                d_24_constrainedPrompt_: _dafny.Seq
                                d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_25_next_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_25_next_ = out17_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_25_next_) == (eosToken):
                                    if (d_2_steps_) < (maxSteps):
                                        d_26_closeBudget_: int
                                        d_26_closeBudget_ = (maxSteps) - (d_2_steps_)
                                        d_27_cg2_: _dafny.Seq
                                        d_28_ci2_: bool
                                        d_29_cc2_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
                                        d_27_cg2_ = out18_
                                        d_28_ci2_ = out19_
                                        d_29_cc2_ = out20_
                                        generated = d_27_cg2_
                                        insideConstrainedOut = d_28_ci2_
                                        currentConstrainedOut = d_29_cc2_
                                        d_2_steps_ = (d_2_steps_) + (d_26_closeBudget_)
                                    raise _dafny.Break("1_1_0")
                                elif True:
                                    d_30_ag_: _dafny.Seq
                                    d_31_ai_: bool
                                    d_32_ac_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                    d_30_ag_ = out21_
                                    d_31_ai_ = out22_
                                    d_32_ac_ = out23_
                                    generated = d_30_ag_
                                    insideConstrainedOut = d_31_ai_
                                    currentConstrainedOut = d_32_ac_
                            pass
                    pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

