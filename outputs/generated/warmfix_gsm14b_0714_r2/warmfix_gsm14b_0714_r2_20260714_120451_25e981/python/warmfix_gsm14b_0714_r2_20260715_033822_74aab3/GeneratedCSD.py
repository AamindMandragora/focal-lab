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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using symbolic variables. Show intermediate calculations in << >> delimiters. The LAST << >> span must contain the complete final answer formula (not a single variable like n or 1)."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        if (d_4_freeStepsTarget_) > (maxSteps):
            d_4_freeStepsTarget_ = maxSteps
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_shouldForce_: bool
                        d_6_shouldForce_ = (not(d_5_forcedFinalSpan_)) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or (((maxSteps) - (d_2_steps_)) <= (6)))
                        if (d_6_shouldForce_) and (((maxSteps) - (d_2_steps_)) >= (3)):
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
                                if (not(d_5_forcedFinalSpan_)) and (((maxSteps) - (d_2_steps_)) >= (3)):
                                    d_11_og_: _dafny.Seq
                                    d_12_oi_: bool
                                    d_13_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_og_ = out4_
                                    d_12_oi_ = out5_
                                    d_13_oc_ = out6_
                                    generated = d_11_og_
                                    insideConstrainedOut = d_12_oi_
                                    currentConstrainedOut = d_13_oc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_5_forcedFinalSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out7_
                                    insideConstrainedOut = out8_
                                    currentConstrainedOut = out9_
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_14_cg_: _dafny.Seq
                            d_15_ci_: bool
                            d_16_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_cg_ = out10_
                            d_15_ci_ = out11_
                            d_16_cc_ = out12_
                            generated = d_14_cg_
                            insideConstrainedOut = d_15_ci_
                            currentConstrainedOut = d_16_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_5_forcedFinalSpan_:
                                raise _dafny.Break("0")
                        elif ((maxSteps) - (d_2_steps_)) <= (4):
                            d_17_closeBudget_: int
                            d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                            d_18_cg_ = out13_
                            d_19_ci_ = out14_
                            d_20_cc_ = out15_
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_2_steps_ = maxSteps
                        elif True:
                            d_21_constrainedPrompt_: _dafny.Seq
                            d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_22_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (d_5_forcedFinalSpan_) and ((len(currentConstrainedOut)) == (0)):
                                d_23_lastTok_: _dafny.Seq
                                d_24_foundLast_: bool
                                out16_: _dafny.Seq
                                out17_: bool
                                out16_, out17_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                                d_23_lastTok_ = out16_
                                d_24_foundLast_ = out17_
                                if d_24_foundLast_:
                                    out18_: _dafny.Seq
                                    out18_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([d_23_lastTok_]), _dafny.BigRational('6e0'), eosToken)
                                    d_22_next_ = out18_
                                elif True:
                                    out19_: _dafny.Seq
                                    out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                    d_22_next_ = out19_
                            elif True:
                                out20_: _dafny.Seq
                                out20_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_22_next_ = out20_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                d_25_remaining_: int
                                d_25_remaining_ = (maxSteps) - (d_2_steps_)
                                if (d_25_remaining_) >= (1):
                                    d_26_cg2_: _dafny.Seq
                                    d_27_ci2_: bool
                                    d_28_cc2_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_remaining_)
                                    d_26_cg2_ = out21_
                                    d_27_ci2_ = out22_
                                    d_28_cc2_ = out23_
                                    generated = d_26_cg2_
                                    insideConstrainedOut = d_27_ci2_
                                    currentConstrainedOut = d_28_cc2_
                                    d_2_steps_ = (d_2_steps_) + (d_25_remaining_)
                                raise _dafny.Break("0")
                            elif True:
                                d_29_ag_: _dafny.Seq
                                d_30_ai_: bool
                                d_31_ac_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: _dafny.Seq
                                out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_29_ag_ = out24_
                                d_30_ai_ = out25_
                                d_31_ac_ = out26_
                                generated = d_29_ag_
                                insideConstrainedOut = d_30_ai_
                                currentConstrainedOut = d_31_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

