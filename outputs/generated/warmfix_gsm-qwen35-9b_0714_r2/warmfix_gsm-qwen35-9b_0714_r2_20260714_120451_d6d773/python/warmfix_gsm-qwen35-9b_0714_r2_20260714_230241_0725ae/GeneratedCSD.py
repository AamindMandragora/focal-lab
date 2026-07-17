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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step using the variable names from the problem. At the very end, write your final answer inside << >> delimiters. CRITICAL RULES: (1) ALWAYS wrap the entire expression in int(): write <<int(expression)>> NOT <<expression>>. (2) Use plain variable names WITHOUT curly braces: write n1 not {n1}, write total not {total}. (3) Do NOT use ** for exponentiation. (4) Allowed operators: +, -, *, /, //, %. (5) Write exactly ONE final <<int(...)>> at the very end. Examples: <<int(n1 * c1 + n2 * c2)>>, <<int(initial - quantity * price)>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_prefixBudget_: int
        if (maxSteps) > (200):
            d_3_prefixBudget_ = (maxSteps) - (80)
        elif True:
            if (maxSteps) > (80):
                d_3_prefixBudget_ = (maxSteps) - (30)
            elif True:
                d_3_prefixBudget_ = _dafny.euclidian_division(maxSteps, 2)
        d_4_penaltyTokens_: _dafny.Seq
        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "**")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\"))])
        d_5_spanTokens_: int
        d_5_spanTokens_ = 0
        d_6_maxSpanTokens_: int
        d_6_maxSpanTokens_ = 28
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_steps_) >= (d_3_prefixBudget_)) and (((maxSteps) - (d_2_steps_)) > (5)):
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
                            d_5_spanTokens_ = 0
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                d_11_eg_: _dafny.Seq
                                d_12_ei_: bool
                                d_13_ec_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_11_eg_ = out4_
                                d_12_ei_ = out5_
                                d_13_ec_ = out6_
                                generated = d_11_eg_
                                insideConstrainedOut = d_12_ei_
                                currentConstrainedOut = d_13_ec_
                                d_5_spanTokens_ = 0
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                    elif True:
                        d_14_remainingBudget_: int
                        d_14_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if (((d_5_spanTokens_) >= (d_6_maxSpanTokens_)) or ((d_14_remainingBudget_) <= (5))) and ((d_14_remainingBudget_) > (0)):
                            d_15_closeBudget_: int
                            if (d_14_remainingBudget_) <= (15):
                                d_15_closeBudget_ = d_14_remainingBudget_
                            elif True:
                                d_15_closeBudget_ = 15
                            d_16_sg_: _dafny.Seq
                            d_17_si_: bool
                            d_18_sc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
                            d_16_sg_ = out7_
                            d_17_si_ = out8_
                            d_18_sc_ = out9_
                            generated = d_16_sg_
                            insideConstrainedOut = d_17_si_
                            currentConstrainedOut = d_18_sc_
                            d_2_steps_ = (d_2_steps_) + (d_15_closeBudget_)
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
                                raise _dafny.Break("0")
                            elif True:
                                d_23_constrainedPrompt_: _dafny.Seq
                                d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_24_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_4_penaltyTokens_, _dafny.BigRational('8e0'), 6, eosToken)
                                d_24_next_ = out14_
                                if (d_24_next_) == (eosToken):
                                    d_25_rem_: int
                                    d_25_rem_ = (maxSteps) - (d_2_steps_)
                                    if (d_25_rem_) > (0):
                                        d_26_closeB_: int
                                        if (d_25_rem_) <= (15):
                                            d_26_closeB_ = d_25_rem_
                                        elif True:
                                            d_26_closeB_ = 15
                                        d_27_sg2_: _dafny.Seq
                                        d_28_si2_: bool
                                        d_29_sc2_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeB_)
                                        d_27_sg2_ = out15_
                                        d_28_si2_ = out16_
                                        d_29_sc2_ = out17_
                                        generated = d_27_sg2_
                                        insideConstrainedOut = d_28_si2_
                                        currentConstrainedOut = d_29_sc2_
                                        d_2_steps_ = (d_2_steps_) + (d_26_closeB_)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_30_ag_: _dafny.Seq
                                    d_31_ai_: bool
                                    d_32_ac_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
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

