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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using symbolic variable names from the problem. Show intermediate calculations inside << >> delimiters. The LAST << >> must contain the complete final symbolic expression as the answer. Do not add any text or spans after the final answer span."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_lastSpanLength_: int
        d_4_lastSpanLength_ = 0
        d_5_minAnswerTokens_: int
        d_5_minAnswerTokens_ = 3
        d_6_reserveBudget_: int
        d_6_reserveBudget_ = 40
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_nearEnd_: bool
                        d_7_nearEnd_ = (((maxSteps) - (d_2_steps_)) <= (d_6_reserveBudget_)) and ((d_2_steps_) > (0))
                        if d_7_nearEnd_:
                            if ((maxSteps) - (d_2_steps_)) >= (2):
                                d_8_og_: _dafny.Seq
                                d_9_oi_: bool
                                d_10_oc_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_8_og_ = out0_
                                d_9_oi_ = out1_
                                d_10_oc_ = out2_
                                generated = d_8_og_
                                insideConstrainedOut = d_9_oi_
                                currentConstrainedOut = d_10_oc_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_2_steps_) < (maxSteps):
                                    d_11_closeBudget_: int
                                    d_11_closeBudget_ = (maxSteps) - (d_2_steps_)
                                    d_12_cg_: _dafny.Seq
                                    d_13_ci_: bool
                                    d_14_cc_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out4_: bool
                                    out5_: _dafny.Seq
                                    out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_11_closeBudget_)
                                    d_12_cg_ = out3_
                                    d_13_ci_ = out4_
                                    d_14_cc_ = out5_
                                    generated = d_12_cg_
                                    insideConstrainedOut = d_13_ci_
                                    currentConstrainedOut = d_14_cc_
                                    d_2_steps_ = (d_2_steps_) + (d_11_closeBudget_)
                            raise _dafny.Break("0")
                        elif True:
                            d_15_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_15_next_ = out6_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                                if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_16_eg_: _dafny.Seq
                                    d_17_ei_: bool
                                    d_18_ec_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_16_eg_ = out7_
                                    d_17_ei_ = out8_
                                    d_18_ec_ = out9_
                                    generated = d_16_eg_
                                    insideConstrainedOut = d_17_ei_
                                    currentConstrainedOut = d_18_ec_
                    elif True:
                        if ((maxSteps) - (d_2_steps_)) <= (5):
                            d_19_closeBudget_: int
                            d_19_closeBudget_ = (maxSteps) - (d_2_steps_)
                            d_20_cg_: _dafny.Seq
                            d_21_ci_: bool
                            d_22_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
                            d_20_cg_ = out10_
                            d_21_ci_ = out11_
                            d_22_cc_ = out12_
                            generated = d_20_cg_
                            insideConstrainedOut = d_21_ci_
                            currentConstrainedOut = d_22_cc_
                            d_2_steps_ = (d_2_steps_) + (d_19_closeBudget_)
                            raise _dafny.Break("0")
                        elif True:
                            d_23_cg_: _dafny.Seq
                            d_24_ci_: bool
                            d_25_cc_: _dafny.Seq
                            d_26_closed_: bool
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out13_, out14_, out15_, out16_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_23_cg_ = out13_
                            d_24_ci_ = out14_
                            d_25_cc_ = out15_
                            d_26_closed_ = out16_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_26_closed_:
                                d_4_lastSpanLength_ = len(currentConstrainedOut)
                                generated = d_23_cg_
                                insideConstrainedOut = d_24_ci_
                                currentConstrainedOut = d_25_cc_
                                if (d_4_lastSpanLength_) >= (d_5_minAnswerTokens_):
                                    raise _dafny.Break("0")
                            elif True:
                                d_27_constrainedPrompt_: _dafny.Seq
                                d_27_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_28_next_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_28_next_ = out17_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_28_next_) == (eosToken):
                                    if (d_2_steps_) < (maxSteps):
                                        d_29_closeBudget_: int
                                        d_29_closeBudget_ = (maxSteps) - (d_2_steps_)
                                        d_30_cg2_: _dafny.Seq
                                        d_31_ci2_: bool
                                        d_32_cc2_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget_)
                                        d_30_cg2_ = out18_
                                        d_31_ci2_ = out19_
                                        d_32_cc2_ = out20_
                                        generated = d_30_cg2_
                                        insideConstrainedOut = d_31_ci2_
                                        currentConstrainedOut = d_32_cc2_
                                        d_2_steps_ = (d_2_steps_) + (d_29_closeBudget_)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_33_ag_: _dafny.Seq
                                    d_34_ai_: bool
                                    d_35_ac_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                                    d_33_ag_ = out21_
                                    d_34_ai_ = out22_
                                    d_35_ac_ = out23_
                                    generated = d_33_ag_
                                    insideConstrainedOut = d_34_ai_
                                    currentConstrainedOut = d_35_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

