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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Use the variable names exactly as they appear in the problem. Do NOT use curly braces in the formula. Write 'price' not 'cur{price}', write 'frac' not '{frac}'. End with exactly one <<int(formula)>>. Use only +, -, *, /, //, % operators. Never use ** or ^. Examples: <<int(n * p)>>, <<int(a - b * c)>>, <<int(total // count)>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_prefixBudget_: int
        if (maxSteps) > (30):
            d_3_prefixBudget_ = (maxSteps) - (30)
        elif True:
            d_3_prefixBudget_ = maxSteps
        d_4_penaltyTokens_: _dafny.Seq
        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "**")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\"))])
        d_5_spanTokens_: int
        d_5_spanTokens_ = 0
        d_6_maxSpanTokens_: int
        d_6_maxSpanTokens_ = 30
        d_7_nearBudgetThreshold_: int
        if (maxSteps) > (50):
            d_7_nearBudgetThreshold_ = 40
        elif True:
            d_7_nearBudgetThreshold_ = _dafny.euclidian_division(maxSteps, 4)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_remainingBudget_: int
                        d_8_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if ((d_2_steps_) >= (d_3_prefixBudget_)) and ((d_8_remainingBudget_) > (0)):
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
                            d_5_spanTokens_ = 0
                            d_12_remAfterOpen_: int
                            d_12_remAfterOpen_ = (maxSteps) - (d_2_steps_)
                            if (d_12_remAfterOpen_) > (0):
                                d_13_sg_: _dafny.Seq
                                d_14_si_: bool
                                d_15_sc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_remAfterOpen_)
                                d_13_sg_ = out3_
                                d_14_si_ = out4_
                                d_15_sc_ = out5_
                                generated = d_13_sg_
                                insideConstrainedOut = d_14_si_
                                currentConstrainedOut = d_15_sc_
                                d_2_steps_ = (d_2_steps_) + (d_12_remAfterOpen_)
                            raise _dafny.Break("0")
                        elif True:
                            d_16_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_16_next_ = out6_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
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
                                d_5_spanTokens_ = 0
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                    elif True:
                        d_20_remainingBudget_: int
                        d_20_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if ((d_20_remainingBudget_) <= (d_7_nearBudgetThreshold_)) or ((d_5_spanTokens_) >= (d_6_maxSpanTokens_)):
                            if (d_20_remainingBudget_) > (0):
                                d_21_sg_: _dafny.Seq
                                d_22_si_: bool
                                d_23_sc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_remainingBudget_)
                                d_21_sg_ = out10_
                                d_22_si_ = out11_
                                d_23_sc_ = out12_
                                generated = d_21_sg_
                                insideConstrainedOut = d_22_si_
                                currentConstrainedOut = d_23_sc_
                                d_2_steps_ = (d_2_steps_) + (d_20_remainingBudget_)
                            raise _dafny.Break("0")
                        d_24_cg_: _dafny.Seq
                        d_25_ci_: bool
                        d_26_cc_: _dafny.Seq
                        d_27_closed_: bool
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out16_: bool
                        out13_, out14_, out15_, out16_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_24_cg_ = out13_
                        d_25_ci_ = out14_
                        d_26_cc_ = out15_
                        d_27_closed_ = out16_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_27_closed_:
                            generated = d_24_cg_
                            insideConstrainedOut = d_25_ci_
                            currentConstrainedOut = d_26_cc_
                            raise _dafny.Break("0")
                        elif True:
                            d_28_constrainedPrompt_: _dafny.Seq
                            d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_29_next_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_4_penaltyTokens_, _dafny.BigRational('8e0'), 6, eosToken)
                            d_29_next_ = out17_
                            if (d_29_next_) == (eosToken):
                                d_30_rem_: int
                                d_30_rem_ = (maxSteps) - (d_2_steps_)
                                if (d_30_rem_) > (0):
                                    d_31_sg2_: _dafny.Seq
                                    d_32_si2_: bool
                                    d_33_sc2_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_rem_)
                                    d_31_sg2_ = out18_
                                    d_32_si2_ = out19_
                                    d_33_sc2_ = out20_
                                    generated = d_31_sg2_
                                    insideConstrainedOut = d_32_si2_
                                    currentConstrainedOut = d_33_sc2_
                                    d_2_steps_ = (d_2_steps_) + (d_30_rem_)
                                raise _dafny.Break("0")
                            elif True:
                                d_34_ag_: _dafny.Seq
                                d_35_ai_: bool
                                d_36_ac_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                                d_34_ag_ = out21_
                                d_35_ai_ = out22_
                                d_36_ac_ = out23_
                                generated = d_34_ag_
                                insideConstrainedOut = d_35_ai_
                                currentConstrainedOut = d_36_ac_
                                d_5_spanTokens_ = (d_5_spanTokens_) + (1)
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

