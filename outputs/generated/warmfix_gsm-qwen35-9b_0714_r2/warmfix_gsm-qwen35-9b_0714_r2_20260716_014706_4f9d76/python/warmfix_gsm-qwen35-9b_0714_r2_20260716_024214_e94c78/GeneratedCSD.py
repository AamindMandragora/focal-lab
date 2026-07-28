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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. End with <<int(formula)>> where formula uses ONLY variable names from the problem (NO curly braces: write n not {n}, frac_1 not {frac_1}). Use only +, -, *, /, //, % operators (never ** or ^). Examples: <<int(n * p)>>, <<int(a + b * c)>>, <<int(total // count)>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_prefixBudget_: int
        if (maxSteps) > (200):
            d_3_prefixBudget_ = (maxSteps) - (100)
        elif True:
            if (maxSteps) > (100):
                d_3_prefixBudget_ = (maxSteps) - (30)
            elif True:
                d_3_prefixBudget_ = _dafny.euclidian_division(maxSteps, 2)
        d_4_penaltyTokens_: _dafny.Seq
        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "**")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\"))])
        d_5_spanTokens_: int
        d_5_spanTokens_ = 0
        d_6_maxSpanTokens_: int
        d_6_maxSpanTokens_ = 22
        d_7_nearBudgetThreshold_: int
        d_7_nearBudgetThreshold_ = 30
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_remainingBudget_: int
                        d_8_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if ((d_2_steps_) >= (d_3_prefixBudget_)) and ((d_8_remainingBudget_) > (5)):
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
                                if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_13_eg_: _dafny.Seq
                                    d_14_ei_: bool
                                    d_15_ec_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_13_eg_ = out4_
                                    d_14_ei_ = out5_
                                    d_15_ec_ = out6_
                                    generated = d_13_eg_
                                    insideConstrainedOut = d_14_ei_
                                    currentConstrainedOut = d_15_ec_
                                    d_5_spanTokens_ = 0
                    elif True:
                        d_16_remainingBudget_: int
                        d_16_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if ((d_16_remainingBudget_) <= (d_7_nearBudgetThreshold_)) or ((d_5_spanTokens_) >= (d_6_maxSpanTokens_)):
                            if (d_16_remainingBudget_) > (0):
                                d_17_sg_: _dafny.Seq
                                d_18_si_: bool
                                d_19_sc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_remainingBudget_)
                                d_17_sg_ = out7_
                                d_18_si_ = out8_
                                d_19_sc_ = out9_
                                generated = d_17_sg_
                                insideConstrainedOut = d_18_si_
                                currentConstrainedOut = d_19_sc_
                                d_2_steps_ = (d_2_steps_) + (d_16_remainingBudget_)
                            raise _dafny.Break("0")
                        d_20_cg_: _dafny.Seq
                        d_21_ci_: bool
                        d_22_cc_: _dafny.Seq
                        d_23_closed_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_20_cg_ = out10_
                        d_21_ci_ = out11_
                        d_22_cc_ = out12_
                        d_23_closed_ = out13_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_23_closed_:
                            generated = d_20_cg_
                            insideConstrainedOut = d_21_ci_
                            currentConstrainedOut = d_22_cc_
                            raise _dafny.Break("0")
                        elif True:
                            d_24_constrainedPrompt_: _dafny.Seq
                            d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_25_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_4_penaltyTokens_, _dafny.BigRational('8e0'), 6, eosToken)
                            d_25_next_ = out14_
                            if (d_25_next_) == (eosToken):
                                d_26_rem_: int
                                d_26_rem_ = (maxSteps) - (d_2_steps_)
                                if (d_26_rem_) > (0):
                                    d_27_sg2_: _dafny.Seq
                                    d_28_si2_: bool
                                    d_29_sc2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_rem_)
                                    d_27_sg2_ = out15_
                                    d_28_si2_ = out16_
                                    d_29_sc2_ = out17_
                                    generated = d_27_sg2_
                                    insideConstrainedOut = d_28_si2_
                                    currentConstrainedOut = d_29_sc2_
                                    d_2_steps_ = (d_2_steps_) + (d_26_rem_)
                                raise _dafny.Break("0")
                            elif True:
                                d_30_ag_: _dafny.Seq
                                d_31_ai_: bool
                                d_32_ac_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                d_30_ag_ = out18_
                                d_31_ai_ = out19_
                                d_32_ac_ = out20_
                                generated = d_30_ag_
                                insideConstrainedOut = d_31_ai_
                                currentConstrainedOut = d_32_ac_
                                d_5_spanTokens_ = (d_5_spanTokens_) + (1)
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

