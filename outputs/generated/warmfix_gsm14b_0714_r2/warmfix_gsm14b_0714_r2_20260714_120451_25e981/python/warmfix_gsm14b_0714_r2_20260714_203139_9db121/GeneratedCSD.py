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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using the specific numbers given in the problem. Substitute actual numeric values at each step. Show intermediate calculations inside << >> and put the final numeric answer in << >>. For example: <<3 * 4>> for an intermediate step, <<12>> for the final answer. Never use placeholder names or {variable} syntax."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_freeStepsTarget_: int
        d_3_freeStepsTarget_ = _dafny.euclidian_division((maxSteps) * (9), 10)
        d_4_forcedFinalSpan_: bool
        d_4_forcedFinalSpan_ = False
        d_5_hasSeenValidSpan_: bool
        d_5_hasSeenValidSpan_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_budgetLeft_: int
                        d_6_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        d_7_shouldForceNow_: bool
                        d_7_shouldForceNow_ = ((not(d_4_forcedFinalSpan_)) and ((d_6_budgetLeft_) <= (5))) and ((d_6_budgetLeft_) >= (3))
                        if d_7_shouldForceNow_:
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
                        elif ((not(d_4_forcedFinalSpan_)) and ((d_2_steps_) >= (d_3_freeStepsTarget_))) and ((d_6_budgetLeft_) >= (3)):
                            d_11_og_: _dafny.Seq
                            d_12_oi_: bool
                            d_13_oc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_11_og_ = out3_
                            d_12_oi_ = out4_
                            d_13_oc_ = out5_
                            generated = d_11_og_
                            insideConstrainedOut = d_12_oi_
                            currentConstrainedOut = d_13_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_4_forcedFinalSpan_ = True
                        elif True:
                            d_14_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_next_ = out6_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                d_15_budgetRemaining_: int
                                d_15_budgetRemaining_ = (maxSteps) - (d_2_steps_)
                                if ((not(d_4_forcedFinalSpan_)) and (not(d_5_hasSeenValidSpan_))) and ((d_15_budgetRemaining_) >= (4)):
                                    d_16_og_: _dafny.Seq
                                    d_17_oi_: bool
                                    d_18_oc_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_16_og_ = out7_
                                    d_17_oi_ = out8_
                                    d_18_oc_ = out9_
                                    generated = d_16_og_
                                    insideConstrainedOut = d_17_oi_
                                    currentConstrainedOut = d_18_oc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_4_forcedFinalSpan_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                                if (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_19_eg_: _dafny.Seq
                                    d_20_ei_: bool
                                    d_21_ec_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_19_eg_ = out10_
                                    d_20_ei_ = out11_
                                    d_21_ec_ = out12_
                                    generated = d_19_eg_
                                    insideConstrainedOut = d_20_ei_
                                    currentConstrainedOut = d_21_ec_
                    elif True:
                        d_22_budgetLeft_: int
                        d_22_budgetLeft_ = (maxSteps) - (d_2_steps_)
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_23_cg_: _dafny.Seq
                            d_24_ci_: bool
                            d_25_cc_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_23_cg_ = out13_
                            d_24_ci_ = out14_
                            d_25_cc_ = out15_
                            generated = d_23_cg_
                            insideConstrainedOut = d_24_ci_
                            currentConstrainedOut = d_25_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_5_hasSeenValidSpan_ = True
                        elif (d_22_budgetLeft_) <= (3):
                            d_26_cg_: _dafny.Seq
                            d_27_ci_: bool
                            d_28_cc_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_budgetLeft_)
                            d_26_cg_ = out16_
                            d_27_ci_ = out17_
                            d_28_cc_ = out18_
                            generated = d_26_cg_
                            insideConstrainedOut = d_27_ci_
                            currentConstrainedOut = d_28_cc_
                            d_2_steps_ = (d_2_steps_) + (d_22_budgetLeft_)
                        elif True:
                            d_29_constrainedPrompt_: _dafny.Seq
                            d_29_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_30_next_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_29_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_30_next_ = out19_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_30_next_) == (eosToken):
                                d_31_budgetRemaining_: int
                                d_31_budgetRemaining_ = (maxSteps) - (d_2_steps_)
                                if (d_31_budgetRemaining_) >= (1):
                                    d_32_cg_: _dafny.Seq
                                    d_33_ci_: bool
                                    d_34_cc_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_31_budgetRemaining_)
                                    d_32_cg_ = out20_
                                    d_33_ci_ = out21_
                                    d_34_cc_ = out22_
                                    generated = d_32_cg_
                                    insideConstrainedOut = d_33_ci_
                                    currentConstrainedOut = d_34_cc_
                                    d_2_steps_ = (d_2_steps_) + (d_31_budgetRemaining_)
                                raise _dafny.Break("0")
                            elif True:
                                d_35_ag_: _dafny.Seq
                                d_36_ai_: bool
                                d_37_ac_: _dafny.Seq
                                out23_: _dafny.Seq
                                out24_: bool
                                out25_: _dafny.Seq
                                out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                d_35_ag_ = out23_
                                d_36_ai_ = out24_
                                d_37_ac_ = out25_
                                generated = d_35_ag_
                                insideConstrainedOut = d_36_ai_
                                currentConstrainedOut = d_37_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

