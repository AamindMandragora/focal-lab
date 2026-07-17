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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. At the very end, write the final answer as a complete symbolic expression inside << >> like <<n*(k+1)/2>>. Use variable names from the problem. The last << >> is the answer."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (8), 10)
        d_5_forcedSpanOpened_: bool
        d_5_forcedSpanOpened_ = False
        d_6_forcedSpanClosed_: bool
        d_6_forcedSpanClosed_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_6_forcedSpanClosed_:
                            raise _dafny.Break("0")
                        d_7_shouldForce_: bool
                        d_7_shouldForce_ = (not(d_5_forcedSpanOpened_)) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or (((maxSteps) - (d_2_steps_)) <= (10)))
                        if (d_7_shouldForce_) and (((maxSteps) - (d_2_steps_)) >= (2)):
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
                            d_5_forcedSpanOpened_ = True
                        elif not(d_5_forcedSpanOpened_):
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                if ((maxSteps) - (d_2_steps_)) >= (2):
                                    d_12_og_: _dafny.Seq
                                    d_13_oi_: bool
                                    d_14_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_12_og_ = out4_
                                    d_13_oi_ = out5_
                                    d_14_oc_ = out6_
                                    generated = d_12_og_
                                    insideConstrainedOut = d_13_oi_
                                    currentConstrainedOut = d_14_oc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_5_forcedSpanOpened_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        if ((maxSteps) - (d_2_steps_)) <= (3):
                            d_15_closeBudget_: int
                            d_15_closeBudget_ = (maxSteps) - (d_2_steps_)
                            d_16_cg_: _dafny.Seq
                            d_17_ci_: bool
                            d_18_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
                            d_16_cg_ = out7_
                            d_17_ci_ = out8_
                            d_18_cc_ = out9_
                            generated = d_16_cg_
                            insideConstrainedOut = d_17_ci_
                            currentConstrainedOut = d_18_cc_
                            d_2_steps_ = (d_2_steps_) + (d_15_closeBudget_)
                            d_6_forcedSpanClosed_ = True
                            raise _dafny.Break("0")
                        elif True:
                            d_19_cg_: _dafny.Seq
                            d_20_ci_: bool
                            d_21_cc_: _dafny.Seq
                            d_22_closed_: bool
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out13_: bool
                            out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_19_cg_ = out10_
                            d_20_ci_ = out11_
                            d_21_cc_ = out12_
                            d_22_closed_ = out13_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_22_closed_:
                                generated = d_19_cg_
                                insideConstrainedOut = d_20_ci_
                                currentConstrainedOut = d_21_cc_
                                d_6_forcedSpanClosed_ = True
                                raise _dafny.Break("0")
                            elif True:
                                d_23_constrainedPrompt_: _dafny.Seq
                                d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_24_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_24_next_ = out14_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_24_next_) == (eosToken):
                                    if (d_2_steps_) < (maxSteps):
                                        d_25_closeBudget_: int
                                        d_25_closeBudget_ = (maxSteps) - (d_2_steps_)
                                        d_26_cg2_: _dafny.Seq
                                        d_27_ci2_: bool
                                        d_28_cc2_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget_)
                                        d_26_cg2_ = out15_
                                        d_27_ci2_ = out16_
                                        d_28_cc2_ = out17_
                                        generated = d_26_cg2_
                                        insideConstrainedOut = d_27_ci2_
                                        currentConstrainedOut = d_28_cc2_
                                        d_2_steps_ = (d_2_steps_) + (d_25_closeBudget_)
                                    d_6_forcedSpanClosed_ = True
                                    raise _dafny.Break("0")
                                elif True:
                                    d_29_ag_: _dafny.Seq
                                    d_30_ai_: bool
                                    d_31_ac_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                    d_29_ag_ = out18_
                                    d_30_ai_ = out19_
                                    d_31_ac_ = out20_
                                    generated = d_29_ag_
                                    insideConstrainedOut = d_30_ai_
                                    currentConstrainedOut = d_31_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

