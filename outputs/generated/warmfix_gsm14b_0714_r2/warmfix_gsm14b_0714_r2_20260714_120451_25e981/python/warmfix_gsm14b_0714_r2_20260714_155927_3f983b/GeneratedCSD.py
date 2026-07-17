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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Compute with the actual values given. Show your work with expressions inside << >> delimiters. End with your final numeric answer inside << >> like <<42>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_forcedFinalSpan_: bool
        d_4_forcedFinalSpan_ = False
        d_5_forcedFinalSpanClosed_: bool
        d_5_forcedFinalSpanClosed_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_5_forcedFinalSpanClosed_:
                            raise _dafny.Break("0")
                        if d_4_forcedFinalSpan_:
                            raise _dafny.Break("0")
                        d_6_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            if (not(d_4_forcedFinalSpan_)) and (((maxSteps) - (d_2_steps_)) >= (2)):
                                d_7_og_: _dafny.Seq
                                d_8_oi_: bool
                                d_9_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_7_og_ = out1_
                                d_8_oi_ = out2_
                                d_9_oc_ = out3_
                                generated = d_7_og_
                                insideConstrainedOut = d_8_oi_
                                currentConstrainedOut = d_9_oc_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_4_forcedFinalSpan_ = True
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_10_eg_: _dafny.Seq
                                d_11_ei_: bool
                                d_12_ec_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_10_eg_ = out4_
                                d_11_ei_ = out5_
                                d_12_ec_ = out6_
                                generated = d_10_eg_
                                insideConstrainedOut = d_11_ei_
                                currentConstrainedOut = d_12_ec_
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_13_cg_: _dafny.Seq
                            d_14_ci_: bool
                            d_15_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_cg_ = out7_
                            d_14_ci_ = out8_
                            d_15_cc_ = out9_
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_4_forcedFinalSpan_:
                                d_5_forcedFinalSpanClosed_ = True
                                raise _dafny.Break("0")
                        elif ((maxSteps) - (d_2_steps_)) <= (4):
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
                            d_2_steps_ = maxSteps
                            if d_4_forcedFinalSpan_:
                                d_5_forcedFinalSpanClosed_ = True
                            raise _dafny.Break("0")
                        elif True:
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_21_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_21_next_ = out13_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                d_22_remainingAfterEos_: int
                                d_22_remainingAfterEos_ = (maxSteps) - (d_2_steps_)
                                if (d_22_remainingAfterEos_) >= (1):
                                    d_23_closeBudget_: int
                                    d_23_closeBudget_ = d_22_remainingAfterEos_
                                    d_24_cg_: _dafny.Seq
                                    d_25_ci_: bool
                                    d_26_cc_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget_)
                                    d_24_cg_ = out14_
                                    d_25_ci_ = out15_
                                    d_26_cc_ = out16_
                                    generated = d_24_cg_
                                    insideConstrainedOut = d_25_ci_
                                    currentConstrainedOut = d_26_cc_
                                    d_2_steps_ = maxSteps
                                    if d_4_forcedFinalSpan_:
                                        d_5_forcedFinalSpanClosed_ = True
                                raise _dafny.Break("0")
                            elif True:
                                d_27_ag_: _dafny.Seq
                                d_28_ai_: bool
                                d_29_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_27_ag_ = out17_
                                d_28_ai_ = out18_
                                d_29_ac_ = out19_
                                generated = d_27_ag_
                                insideConstrainedOut = d_28_ai_
                                currentConstrainedOut = d_29_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

