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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the given symbolic variable names. Do NOT use {variable} template notation. At the very end, output ONLY the final answer inside the LAST << >> delimiters as a simplified arithmetic expression using the variable names directly (e.g., <<n - int(n * frac)>> or <<total * fraction - current>>)."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_naturalSpanTokens_: int
        d_6_naturalSpanTokens_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_shouldForce_: bool
                        d_7_shouldForce_ = (not(d_5_forcedFinalSpan_)) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or (((maxSteps) - (d_2_steps_)) <= (5)))
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
                            d_5_forcedFinalSpan_ = True
                            d_6_naturalSpanTokens_ = 0
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                if (not(d_5_forcedFinalSpan_)) and (((maxSteps) - (d_2_steps_)) >= (2)):
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
                                    d_5_forcedFinalSpan_ = True
                                    d_6_naturalSpanTokens_ = 0
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
                                    d_6_naturalSpanTokens_ = 0
                    elif True:
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and (d_5_forcedFinalSpan_):
                            d_15_cg_: _dafny.Seq
                            d_16_ci_: bool
                            d_17_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_15_cg_ = out10_
                            d_16_ci_ = out11_
                            d_17_cc_ = out12_
                            generated = d_15_cg_
                            insideConstrainedOut = d_16_ci_
                            currentConstrainedOut = d_17_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif ((parser).IsCompletePrefix(currentConstrainedOut)) and (not(d_5_forcedFinalSpan_)):
                            d_18_naturalLen_: int
                            d_18_naturalLen_ = len(currentConstrainedOut)
                            d_19_cg_: _dafny.Seq
                            d_20_ci_: bool
                            d_21_cc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_cg_ = out13_
                            d_20_ci_ = out14_
                            d_21_cc_ = out15_
                            generated = d_19_cg_
                            insideConstrainedOut = d_20_ci_
                            currentConstrainedOut = d_21_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_6_naturalSpanTokens_ = d_18_naturalLen_
                            if (d_18_naturalLen_) >= (4):
                                d_5_forcedFinalSpan_ = True
                        elif ((maxSteps) - (d_2_steps_)) <= (4):
                            d_22_closeBudget_: int
                            d_22_closeBudget_ = (maxSteps) - (d_2_steps_)
                            d_23_cg_: _dafny.Seq
                            d_24_ci_: bool
                            d_25_cc_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
                            d_23_cg_ = out16_
                            d_24_ci_ = out17_
                            d_25_cc_ = out18_
                            generated = d_23_cg_
                            insideConstrainedOut = d_24_ci_
                            currentConstrainedOut = d_25_cc_
                            d_2_steps_ = maxSteps
                        elif True:
                            d_26_constrainedPrompt_: _dafny.Seq
                            d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_27_next_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_27_next_ = out19_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_27_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_28_cg_: _dafny.Seq
                                    d_29_ci_: bool
                                    d_30_cc_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_28_cg_ = out20_
                                    d_29_ci_ = out21_
                                    d_30_cc_ = out22_
                                    generated = d_28_cg_
                                    insideConstrainedOut = d_29_ci_
                                    currentConstrainedOut = d_30_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    if d_5_forcedFinalSpan_:
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_31_naturalLen_: int
                                        d_31_naturalLen_ = len(d_30_cc_)
                                        if (d_31_naturalLen_) == (0):
                                            d_6_naturalSpanTokens_ = len(currentConstrainedOut)
                                            if (d_6_naturalSpanTokens_) >= (4):
                                                d_5_forcedFinalSpan_ = True
                                elif (d_2_steps_) < (maxSteps):
                                    d_32_closeBudget_: int
                                    d_32_closeBudget_ = (maxSteps) - (d_2_steps_)
                                    d_33_cg_: _dafny.Seq
                                    d_34_ci_: bool
                                    d_35_cc_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudget_)
                                    d_33_cg_ = out23_
                                    d_34_ci_ = out24_
                                    d_35_cc_ = out25_
                                    generated = d_33_cg_
                                    insideConstrainedOut = d_34_ci_
                                    currentConstrainedOut = d_35_cc_
                                    d_2_steps_ = maxSteps
                                raise _dafny.Break("0")
                            elif True:
                                d_36_ag_: _dafny.Seq
                                d_37_ai_: bool
                                d_38_ac_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                                d_36_ag_ = out26_
                                d_37_ai_ = out27_
                                d_38_ac_ = out28_
                                generated = d_36_ag_
                                insideConstrainedOut = d_37_ai_
                                currentConstrainedOut = d_38_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

