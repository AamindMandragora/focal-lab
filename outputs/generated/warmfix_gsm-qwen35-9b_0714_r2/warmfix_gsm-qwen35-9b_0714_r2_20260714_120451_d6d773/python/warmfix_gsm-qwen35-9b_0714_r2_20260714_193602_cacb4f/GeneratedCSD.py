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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step using the variable names from the problem. At the very end, write your final answer inside << >> delimiters. CRITICAL RULES: (1) ALWAYS wrap the entire expression in int(): write <<int(expression)>> NOT <<expression>>. (2) Use plain variable names WITHOUT curly braces: write n1 not {n1}, write total not {total}. (3) Do NOT use ** for exponentiation - use int() for integer conversion only. (4) Allowed operators: +, -, *, /, //, %. (5) Write exactly ONE final <<int(...)>> at the very end. Examples: <<int(n1 * c1 + n2 * c2)>>, <<int(initial - quantity * price)>>, <<int(n * frac1 * mult)>>."))
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
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_steps_) >= (d_3_prefixBudget_)) and (((maxSteps) - (d_2_steps_)) > (5)):
                            d_5_og_: _dafny.Seq
                            d_6_oi_: bool
                            d_7_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_og_ = out0_
                            d_6_oi_ = out1_
                            d_7_oc_ = out2_
                            generated = d_5_og_
                            insideConstrainedOut = d_6_oi_
                            currentConstrainedOut = d_7_oc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                d_9_eg_: _dafny.Seq
                                d_10_ei_: bool
                                d_11_ec_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_9_eg_ = out4_
                                d_10_ei_ = out5_
                                d_11_ec_ = out6_
                                generated = d_9_eg_
                                insideConstrainedOut = d_10_ei_
                                currentConstrainedOut = d_11_ec_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                    elif True:
                        d_12_remainingBudget_: int
                        d_12_remainingBudget_ = (maxSteps) - (d_2_steps_)
                        if ((d_12_remainingBudget_) <= (5)) and ((d_12_remainingBudget_) > (0)):
                            d_13_sg_: _dafny.Seq
                            d_14_si_: bool
                            d_15_sc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_12_remainingBudget_)
                            d_13_sg_ = out7_
                            d_14_si_ = out8_
                            d_15_sc_ = out9_
                            generated = d_13_sg_
                            insideConstrainedOut = d_14_si_
                            currentConstrainedOut = d_15_sc_
                            d_2_steps_ = (d_2_steps_) + (d_12_remainingBudget_)
                            raise _dafny.Break("0")
                        elif True:
                            d_16_cg_: _dafny.Seq
                            d_17_ci_: bool
                            d_18_cc_: _dafny.Seq
                            d_19_closed_: bool
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out13_: bool
                            out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_16_cg_ = out10_
                            d_17_ci_ = out11_
                            d_18_cc_ = out12_
                            d_19_closed_ = out13_
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
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_4_penaltyTokens_, _dafny.BigRational('8e0'), 6, eosToken)
                                d_21_next_ = out14_
                                if (d_21_next_) == (eosToken):
                                    d_22_rem_: int
                                    d_22_rem_ = (maxSteps) - (d_2_steps_)
                                    if (d_22_rem_) > (0):
                                        d_23_sg2_: _dafny.Seq
                                        d_24_si2_: bool
                                        d_25_sc2_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_rem_)
                                        d_23_sg2_ = out15_
                                        d_24_si2_ = out16_
                                        d_25_sc2_ = out17_
                                        generated = d_23_sg2_
                                        insideConstrainedOut = d_24_si2_
                                        currentConstrainedOut = d_25_sc2_
                                        d_2_steps_ = (d_2_steps_) + (d_22_rem_)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_26_ag_: _dafny.Seq
                                    d_27_ai_: bool
                                    d_28_ac_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_26_ag_ = out18_
                                    d_27_ai_ = out19_
                                    d_28_ac_ = out20_
                                    generated = d_26_ag_
                                    insideConstrainedOut = d_27_ai_
                                    currentConstrainedOut = d_28_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

