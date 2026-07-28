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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using plain text only. Do NOT use << >> delimiters anywhere in your reasoning. After completing your reasoning, you will be prompted to provide the final answer in a specific format."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_freeStepsTarget_: int
        d_4_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (65), 100)
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        d_6_longNaturalSpanSeen_: bool
        d_6_longNaturalSpanSeen_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_budgetRemaining_: int
                        d_7_budgetRemaining_ = (maxSteps) - (d_2_steps_)
                        d_8_shouldForce_: bool
                        d_8_shouldForce_ = ((not(d_5_forcedFinalSpan_)) and (not(d_6_longNaturalSpanSeen_))) and (((d_2_steps_) >= (d_4_freeStepsTarget_)) or ((d_7_budgetRemaining_) <= (5)))
                        if (d_8_shouldForce_) and ((d_7_budgetRemaining_) >= (2)):
                            d_9_og_: _dafny.Seq
                            d_10_oi_: bool
                            d_11_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_og_ = out0_
                            d_10_oi_ = out1_
                            d_11_oc_ = out2_
                            generated = d_9_og_
                            insideConstrainedOut = d_10_oi_
                            currentConstrainedOut = d_11_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_5_forcedFinalSpan_ = True
                        elif d_6_longNaturalSpanSeen_:
                            raise _dafny.Break("0")
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out4_
                                    insideConstrainedOut = out5_
                                    currentConstrainedOut = out6_
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_13_spanLen_: int
                            d_13_spanLen_ = len(currentConstrainedOut)
                            d_14_cg_: _dafny.Seq
                            d_15_ci_: bool
                            d_16_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_cg_ = out7_
                            d_15_ci_ = out8_
                            d_16_cc_ = out9_
                            generated = d_14_cg_
                            insideConstrainedOut = d_15_ci_
                            currentConstrainedOut = d_16_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_5_forcedFinalSpan_:
                                raise _dafny.Break("0")
                            if (d_13_spanLen_) >= (2):
                                d_6_longNaturalSpanSeen_ = True
                                raise _dafny.Break("0")
                        elif ((maxSteps) - (d_2_steps_)) <= (3):
                            d_17_closeBudget_: int
                            d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
                            d_18_cg_: _dafny.Seq
                            d_19_ci_: bool
                            d_20_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                            d_18_cg_ = out10_
                            d_19_ci_ = out11_
                            d_20_cc_ = out12_
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            d_2_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            d_21_constrainedPrompt_: _dafny.Seq
                            d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_22_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                            d_22_next_ = out13_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_23_cg_: _dafny.Seq
                                    d_24_ci_: bool
                                    d_25_cc_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_23_cg_ = out14_
                                    d_24_ci_ = out15_
                                    d_25_cc_ = out16_
                                    generated = d_23_cg_
                                    insideConstrainedOut = d_24_ci_
                                    currentConstrainedOut = d_25_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                elif (d_2_steps_) < (maxSteps):
                                    d_26_closeBudget_: int
                                    d_26_closeBudget_ = (maxSteps) - (d_2_steps_)
                                    d_27_cg_: _dafny.Seq
                                    d_28_ci_: bool
                                    d_29_cc_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
                                    d_27_cg_ = out17_
                                    d_28_ci_ = out18_
                                    d_29_cc_ = out19_
                                    generated = d_27_cg_
                                    insideConstrainedOut = d_28_ci_
                                    currentConstrainedOut = d_29_cc_
                                    d_2_steps_ = maxSteps
                                raise _dafny.Break("0")
                            elif True:
                                d_30_ag_: _dafny.Seq
                                d_31_ai_: bool
                                d_32_ac_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
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

