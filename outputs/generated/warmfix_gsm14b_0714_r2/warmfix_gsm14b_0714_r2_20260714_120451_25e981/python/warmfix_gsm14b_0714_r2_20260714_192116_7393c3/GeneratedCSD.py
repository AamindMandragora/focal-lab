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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the SPECIFIC NUMERIC VALUES given in the problem. Do NOT use template variable names like {n} or {name}. Compute actual numbers at each step. Place intermediate results and the final numeric answer inside << >> delimiters. Example: <<6 * 7>> for intermediate, <<42>> for final answer."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_spanTokenCount_: int
        d_4_spanTokenCount_ = 0
        d_5_minSpanTokens_: int
        d_5_minSpanTokens_ = 3
        d_6_forcedFinalSpan_: bool
        d_6_forcedFinalSpan_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_budgetLeft_: int
                        d_7_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        d_8_shouldForce_: bool
                        d_8_shouldForce_ = ((not(d_6_forcedFinalSpan_)) and ((d_7_budgetLeft_) <= (10))) and ((d_7_budgetLeft_) >= (3))
                        if d_8_shouldForce_:
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
                            d_6_forcedFinalSpan_ = True
                            d_4_spanTokenCount_ = 0
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                d_13_budgetRemaining_: int
                                d_13_budgetRemaining_ = (maxSteps) - (d_2_steps_)
                                if (not(d_6_forcedFinalSpan_)) and ((d_13_budgetRemaining_) >= (4)):
                                    d_14_og_: _dafny.Seq
                                    d_15_oi_: bool
                                    d_16_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_14_og_ = out4_
                                    d_15_oi_ = out5_
                                    d_16_oc_ = out6_
                                    generated = d_14_og_
                                    insideConstrainedOut = d_15_oi_
                                    currentConstrainedOut = d_16_oc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_6_forcedFinalSpan_ = True
                                    d_4_spanTokenCount_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_17_eg_: _dafny.Seq
                                    d_18_ei_: bool
                                    d_19_ec_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_17_eg_ = out7_
                                    d_18_ei_ = out8_
                                    d_19_ec_ = out9_
                                    generated = d_17_eg_
                                    insideConstrainedOut = d_18_ei_
                                    currentConstrainedOut = d_19_ec_
                                    d_4_spanTokenCount_ = 0
                    elif True:
                        d_20_budgetLeft_: int
                        d_20_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        d_21_spanLongEnough_: bool
                        d_21_spanLongEnough_ = ((d_4_spanTokenCount_) >= (d_5_minSpanTokens_)) or ((len(currentConstrainedOut)) >= (d_5_minSpanTokens_))
                        if (d_21_spanLongEnough_) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_22_cg_: _dafny.Seq
                            d_23_ci_: bool
                            d_24_cc_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_22_cg_ = out10_
                            d_23_ci_ = out11_
                            d_24_cc_ = out12_
                            generated = d_22_cg_
                            insideConstrainedOut = d_23_ci_
                            currentConstrainedOut = d_24_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_4_spanTokenCount_ = 0
                        elif (d_20_budgetLeft_) <= (4):
                            d_25_closeBudget_: int
                            d_25_closeBudget_ = d_20_budgetLeft_
                            d_26_cg_: _dafny.Seq
                            d_27_ci_: bool
                            d_28_cc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget_)
                            d_26_cg_ = out13_
                            d_27_ci_ = out14_
                            d_28_cc_ = out15_
                            generated = d_26_cg_
                            insideConstrainedOut = d_27_ci_
                            currentConstrainedOut = d_28_cc_
                            d_2_steps_ = maxSteps
                        elif True:
                            d_29_constrainedPrompt_: _dafny.Seq
                            d_29_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_30_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_30_next_ = out16_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_30_next_) == (eosToken):
                                d_31_budgetRemaining_: int
                                d_31_budgetRemaining_ = (maxSteps) - (d_2_steps_)
                                if (d_31_budgetRemaining_) >= (1):
                                    d_32_closeBudget_: int
                                    d_32_closeBudget_ = d_31_budgetRemaining_
                                    d_33_cg_: _dafny.Seq
                                    d_34_ci_: bool
                                    d_35_cc_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_32_closeBudget_)
                                    d_33_cg_ = out17_
                                    d_34_ci_ = out18_
                                    d_35_cc_ = out19_
                                    generated = d_33_cg_
                                    insideConstrainedOut = d_34_ci_
                                    currentConstrainedOut = d_35_cc_
                                    d_2_steps_ = maxSteps
                                raise _dafny.Break("0")
                            elif True:
                                d_36_ag_: _dafny.Seq
                                d_37_ai_: bool
                                d_38_ac_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                d_36_ag_ = out20_
                                d_37_ai_ = out21_
                                d_38_ac_ = out22_
                                generated = d_36_ag_
                                insideConstrainedOut = d_37_ai_
                                currentConstrainedOut = d_38_ac_
                                d_4_spanTokenCount_ = (d_4_spanTokenCount_) + (1)
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

