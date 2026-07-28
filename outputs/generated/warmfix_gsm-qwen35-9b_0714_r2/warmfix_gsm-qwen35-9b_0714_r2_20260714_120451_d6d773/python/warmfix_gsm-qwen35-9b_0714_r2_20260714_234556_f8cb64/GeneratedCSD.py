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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. At the very end, write your final symbolic answer inside << >> delimiters. IMPORTANT: (1) Wrap the final expression in int(): <<int(expression)>>. (2) Use plain variable names WITHOUT curly braces: write n1 not {n1}, write cur not {cur}. (3) Do NOT use ** for exponentiation in the final expression. (4) Allowed operators: +, -, *, /, //, %. (5) Write exactly ONE final <<int(expression)>> spanning a complete arithmetic expression. Example: <<int(n1 * p1 + n2 * p2)>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "**")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\"))])
        d_4_spanTokens_: int
        d_4_spanTokens_ = 0
        d_5_maxSpanTokens_: int
        d_5_maxSpanTokens_ = 35
        d_6_closeReserve_: int
        d_6_closeReserve_ = 20
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_7_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                            d_8_eg_: _dafny.Seq
                            d_9_ei_: bool
                            d_10_ec_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_eg_ = out1_
                            d_9_ei_ = out2_
                            d_10_ec_ = out3_
                            generated = d_8_eg_
                            insideConstrainedOut = d_9_ei_
                            currentConstrainedOut = d_10_ec_
                            d_4_spanTokens_ = 0
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                    elif True:
                        d_11_remainingBudget_: int
                        d_11_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if ((d_4_spanTokens_) >= (d_5_maxSpanTokens_)) or ((d_11_remainingBudget_) <= (d_6_closeReserve_)):
                            d_12_closeBudget_: int
                            if (d_11_remainingBudget_) <= (d_6_closeReserve_):
                                d_12_closeBudget_ = d_11_remainingBudget_
                            elif True:
                                d_12_closeBudget_ = d_6_closeReserve_
                            d_13_sg_: _dafny.Seq
                            d_14_si_: bool
                            d_15_sc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_closeBudget_)
                            d_13_sg_ = out4_
                            d_14_si_ = out5_
                            d_15_sc_ = out6_
                            generated = d_13_sg_
                            insideConstrainedOut = d_14_si_
                            currentConstrainedOut = d_15_sc_
                            d_2_steps_ = (d_2_steps_) + (d_12_closeBudget_)
                            raise _dafny.Break("0")
                        d_16_cg_: _dafny.Seq
                        d_17_ci_: bool
                        d_18_cc_: _dafny.Seq
                        d_19_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_16_cg_ = out7_
                        d_17_ci_ = out8_
                        d_18_cc_ = out9_
                        d_19_closed_ = out10_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_19_closed_:
                            generated = d_16_cg_
                            insideConstrainedOut = d_17_ci_
                            currentConstrainedOut = d_18_cc_
                            raise _dafny.Break("0")
                        elif True:
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_21_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_3_penaltyTokens_, _dafny.BigRational('8e0'), 6, eosToken)
                            d_21_next_ = out11_
                            if (d_21_next_) == (eosToken):
                                d_22_rem_: int
                                d_22_rem_ = (maxSteps) - (d_2_steps_)
                                if (d_22_rem_) > (0):
                                    d_23_closeB_: int
                                    if (d_22_rem_) <= (d_6_closeReserve_):
                                        d_23_closeB_ = d_22_rem_
                                    elif True:
                                        d_23_closeB_ = d_6_closeReserve_
                                    d_24_sg2_: _dafny.Seq
                                    d_25_si2_: bool
                                    d_26_sc2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeB_)
                                    d_24_sg2_ = out12_
                                    d_25_si2_ = out13_
                                    d_26_sc2_ = out14_
                                    generated = d_24_sg2_
                                    insideConstrainedOut = d_25_si2_
                                    currentConstrainedOut = d_26_sc2_
                                    d_2_steps_ = (d_2_steps_) + (d_23_closeB_)
                                raise _dafny.Break("0")
                            elif True:
                                d_27_ag_: _dafny.Seq
                                d_28_ai_: bool
                                d_29_ac_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_27_ag_ = out15_
                                d_28_ai_ = out16_
                                d_29_ac_ = out17_
                                generated = d_27_ag_
                                insideConstrainedOut = d_28_ai_
                                currentConstrainedOut = d_29_ac_
                                d_4_spanTokens_ = (d_4_spanTokens_) + (1)
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

