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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. At the end, output the final arithmetic expression inside << >> delimiters. Use plain variable names (no curly braces). Use // for integer division. Do not use **. Example: <<(a + b) * c>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (7), 10)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_minSpanTokens_: int
        d_6_minSpanTokens_ = 3
        d_7_maxSpanTokens_: int
        d_7_maxSpanTokens_ = 50
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_remainingBudget_: int
                        d_8_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        d_9_shouldForce_: bool
                        d_9_shouldForce_ = (not(d_5_forcedFinalSpan_)) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or ((d_8_remainingBudget_) <= (6)))
                        if (d_9_shouldForce_) and ((d_8_remainingBudget_) >= (3)):
                            d_10_og_: _dafny.Seq
                            d_11_oi_: bool
                            d_12_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_og_ = out0_
                            d_11_oi_ = out1_
                            d_12_oc_ = out2_
                            generated = d_10_og_
                            insideConstrainedOut = d_11_oi_
                            currentConstrainedOut = d_12_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_5_forcedFinalSpan_ = True
                        elif True:
                            d_13_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                if (not(d_5_forcedFinalSpan_)) and (((maxSteps) - (d_2_steps_)) >= (3)):
                                    d_14_og2_: _dafny.Seq
                                    d_15_oi2_: bool
                                    d_16_oc2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_14_og2_ = out4_
                                    d_15_oi2_ = out5_
                                    d_16_oc2_ = out6_
                                    generated = d_14_og2_
                                    insideConstrainedOut = d_15_oi2_
                                    currentConstrainedOut = d_16_oc2_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_5_forcedFinalSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out7_
                                    insideConstrainedOut = out8_
                                    currentConstrainedOut = out9_
                    elif True:
                        d_17_remainingBudget2_: int
                        d_17_remainingBudget2_ = (maxSteps) - (d_2_steps_)
                        d_18_spanLen_: int
                        d_18_spanLen_ = len(currentConstrainedOut)
                        if (d_18_spanLen_) >= (d_7_maxSpanTokens_):
                            if (d_17_remainingBudget2_) >= (1):
                                d_19_closeBudget_: int
                                d_19_closeBudget_ = d_17_remainingBudget2_
                                if (d_19_closeBudget_) > (5):
                                    d_19_closeBudget_ = 5
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
                                if (d_2_steps_) > (maxSteps):
                                    d_2_steps_ = maxSteps
                            if d_5_forcedFinalSpan_:
                                raise _dafny.Break("0")
                        elif (d_17_remainingBudget2_) <= (3):
                            d_23_closeBudget2_: int
                            d_23_closeBudget2_ = d_17_remainingBudget2_
                            d_24_cg2_: _dafny.Seq
                            d_25_ci2_: bool
                            d_26_cc2_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget2_)
                            d_24_cg2_ = out13_
                            d_25_ci2_ = out14_
                            d_26_cc2_ = out15_
                            generated = d_24_cg2_
                            insideConstrainedOut = d_25_ci2_
                            currentConstrainedOut = d_26_cc2_
                            d_2_steps_ = (d_2_steps_) + (d_23_closeBudget2_)
                            if (d_2_steps_) > (maxSteps):
                                d_2_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            if (d_18_spanLen_) >= (d_6_minSpanTokens_):
                                d_27_cg3_: _dafny.Seq
                                d_28_ci3_: bool
                                d_29_cc3_: _dafny.Seq
                                d_30_closed_: bool
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out19_: bool
                                out16_, out17_, out18_, out19_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_27_cg3_ = out16_
                                d_28_ci3_ = out17_
                                d_29_cc3_ = out18_
                                d_30_closed_ = out19_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if d_30_closed_:
                                    generated = d_27_cg3_
                                    insideConstrainedOut = d_28_ci3_
                                    currentConstrainedOut = d_29_cc3_
                                    if d_5_forcedFinalSpan_:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_31_constrainedPrompt_: _dafny.Seq
                                    d_31_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_32_next2_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out20_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_31_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                    d_32_next2_ = out20_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    if (d_32_next2_) == (eosToken):
                                        if ((maxSteps) - (d_2_steps_)) >= (2):
                                            d_33_closeBudget3_: int
                                            d_33_closeBudget3_ = (maxSteps) - (d_2_steps_)
                                            if (d_33_closeBudget3_) > (5):
                                                d_33_closeBudget3_ = 5
                                            d_34_cg4_: _dafny.Seq
                                            d_35_ci4_: bool
                                            d_36_cc4_: _dafny.Seq
                                            out21_: _dafny.Seq
                                            out22_: bool
                                            out23_: _dafny.Seq
                                            out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_closeBudget3_)
                                            d_34_cg4_ = out21_
                                            d_35_ci4_ = out22_
                                            d_36_cc4_ = out23_
                                            generated = d_34_cg4_
                                            insideConstrainedOut = d_35_ci4_
                                            currentConstrainedOut = d_36_cc4_
                                            d_2_steps_ = (d_2_steps_) + (d_33_closeBudget3_)
                                            if (d_2_steps_) > (maxSteps):
                                                d_2_steps_ = maxSteps
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_37_ag_: _dafny.Seq
                                        d_38_ai_: bool
                                        d_39_ac_: _dafny.Seq
                                        out24_: _dafny.Seq
                                        out25_: bool
                                        out26_: _dafny.Seq
                                        out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next2_)
                                        d_37_ag_ = out24_
                                        d_38_ai_ = out25_
                                        d_39_ac_ = out26_
                                        generated = d_37_ag_
                                        insideConstrainedOut = d_38_ai_
                                        currentConstrainedOut = d_39_ac_
                            elif True:
                                d_40_constrainedPrompt2_: _dafny.Seq
                                d_40_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_41_next3_: _dafny.Seq
                                out27_: _dafny.Seq
                                out27_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_40_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                d_41_next3_ = out27_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_41_next3_) == (eosToken):
                                    if ((maxSteps) - (d_2_steps_)) >= (2):
                                        d_42_closeBudget4_: int
                                        d_42_closeBudget4_ = (maxSteps) - (d_2_steps_)
                                        if (d_42_closeBudget4_) > (5):
                                            d_42_closeBudget4_ = 5
                                        d_43_cg5_: _dafny.Seq
                                        d_44_ci5_: bool
                                        d_45_cc5_: _dafny.Seq
                                        out28_: _dafny.Seq
                                        out29_: bool
                                        out30_: _dafny.Seq
                                        out28_, out29_, out30_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_42_closeBudget4_)
                                        d_43_cg5_ = out28_
                                        d_44_ci5_ = out29_
                                        d_45_cc5_ = out30_
                                        generated = d_43_cg5_
                                        insideConstrainedOut = d_44_ci5_
                                        currentConstrainedOut = d_45_cc5_
                                        d_2_steps_ = (d_2_steps_) + (d_42_closeBudget4_)
                                        if (d_2_steps_) > (maxSteps):
                                            d_2_steps_ = maxSteps
                                    raise _dafny.Break("0")
                                elif True:
                                    d_46_ag2_: _dafny.Seq
                                    d_47_ai2_: bool
                                    d_48_ac2_: _dafny.Seq
                                    out31_: _dafny.Seq
                                    out32_: bool
                                    out33_: _dafny.Seq
                                    out31_, out32_, out33_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_41_next3_)
                                    d_46_ag2_ = out31_
                                    d_47_ai2_ = out32_
                                    d_48_ac2_ = out33_
                                    generated = d_46_ag2_
                                    insideConstrainedOut = d_47_ai2_
                                    currentConstrainedOut = d_48_ac2_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

