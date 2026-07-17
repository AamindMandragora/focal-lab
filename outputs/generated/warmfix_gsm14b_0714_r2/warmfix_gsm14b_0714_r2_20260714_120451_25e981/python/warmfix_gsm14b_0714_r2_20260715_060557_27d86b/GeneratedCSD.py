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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. Use << >> delimiters for each calculation step. The LAST << >> span must contain the complete final formula using the variable names from the problem (e.g., <<n * rate * days>> or <<price * (1 + tax)>>). Do not summarize with just a variable name."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_forcedFinalSpan_: bool
        d_4_forcedFinalSpan_ = False
        d_5_lastCompletedSpanSize_: int
        d_5_lastCompletedSpanSize_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_nearEnd_: bool
                        d_6_nearEnd_ = ((maxSteps) - (d_2_steps_)) <= (3)
                        if d_6_nearEnd_:
                            raise _dafny.Break("0")
                        d_7_shouldForce_: bool
                        d_7_shouldForce_ = (not(d_4_forcedFinalSpan_)) and ((d_2_steps_) >= (_dafny.euclidian_division((maxSteps) * (7), 10)))
                        if (d_7_shouldForce_) and (((maxSteps) - (d_2_steps_)) >= (3)):
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
                            d_4_forcedFinalSpan_ = True
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                d_12_needForce_: bool
                                d_12_needForce_ = ((not(d_4_forcedFinalSpan_)) and ((d_5_lastCompletedSpanSize_) <= (2))) and (((maxSteps) - (d_2_steps_)) >= (3))
                                if d_12_needForce_:
                                    d_13_og_: _dafny.Seq
                                    d_14_oi_: bool
                                    d_15_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_13_og_ = out4_
                                    d_14_oi_ = out5_
                                    d_15_oc_ = out6_
                                    generated = d_13_og_
                                    insideConstrainedOut = d_14_oi_
                                    currentConstrainedOut = d_15_oc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_4_forcedFinalSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out7_
                                    insideConstrainedOut = out8_
                                    currentConstrainedOut = out9_
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_5_lastCompletedSpanSize_ = len(currentConstrainedOut)
                            d_16_cg_: _dafny.Seq
                            d_17_ci_: bool
                            d_18_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_cg_ = out10_
                            d_17_ci_ = out11_
                            d_18_cc_ = out12_
                            generated = d_16_cg_
                            insideConstrainedOut = d_17_ci_
                            currentConstrainedOut = d_18_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_4_forcedFinalSpan_:
                                raise _dafny.Break("0")
                        elif ((maxSteps) - (d_2_steps_)) <= (4):
                            d_19_closeBudget_: int
                            d_19_closeBudget_ = (maxSteps) - (d_2_steps_)
                            d_20_cg_: _dafny.Seq
                            d_21_ci_: bool
                            d_22_cc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
                            d_20_cg_ = out13_
                            d_21_ci_ = out14_
                            d_22_cc_ = out15_
                            generated = d_20_cg_
                            insideConstrainedOut = d_21_ci_
                            currentConstrainedOut = d_22_cc_
                            d_2_steps_ = maxSteps
                        elif True:
                            d_23_constrainedPrompt_: _dafny.Seq
                            d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_24_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_24_next_ = out16_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_24_next_) == (eosToken):
                                d_25_remaining_: int
                                d_25_remaining_ = (maxSteps) - (d_2_steps_)
                                if (d_25_remaining_) >= (1):
                                    d_26_closeBudget_: int
                                    d_26_closeBudget_ = d_25_remaining_
                                    d_27_cg2_: _dafny.Seq
                                    d_28_ci2_: bool
                                    d_29_cc2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
                                    d_27_cg2_ = out17_
                                    d_28_ci2_ = out18_
                                    d_29_cc2_ = out19_
                                    generated = d_27_cg2_
                                    insideConstrainedOut = d_28_ci2_
                                    currentConstrainedOut = d_29_cc2_
                                    d_2_steps_ = (d_2_steps_) + (d_26_closeBudget_)
                                raise _dafny.Break("0")
                            elif True:
                                d_30_ag_: _dafny.Seq
                                d_31_ai_: bool
                                d_32_ac_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                d_30_ag_ = out20_
                                d_31_ai_ = out21_
                                d_32_ac_ = out22_
                                generated = d_30_ag_
                                insideConstrainedOut = d_31_ai_
                                currentConstrainedOut = d_32_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

